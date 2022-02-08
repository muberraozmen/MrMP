import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from network.Encoder import *
from network.Decoder import *
from network.CompGCN import *
from network.Modules import XavierLinear, XavierEmbedding

__all__ = ['MrMP']


class MrMP(nn.Module):

    def __init__(self, opt, adjs=None):

        super().__init__()

        self.src_word_emb = XavierEmbedding((opt.n_src_vocab, opt.d_model), padding_idx=0)
        self.encoder = Encoder(opt)

        self.tgt_array = torch.from_numpy(np.arange(opt.n_tgt_vocab)).view(-1, 1).transpose(0, 1)
        if opt.cuda_on is True:
            self.tgt_array = self.tgt_array.cuda()
        self.tgt_word_emb = XavierEmbedding((opt.n_tgt_vocab, opt.d_model), padding_idx=0)

        if opt.mrmp_on is True:
            self.num_relns = len(adjs)
            self.reln_array = torch.from_numpy(np.arange(self.num_relns * 3))  # x3 in-out-loop
            if opt.cuda_on is True:
                self.reln_array = self.reln_array.cuda()
            self.reln_order_emb = XavierEmbedding((self.num_relns * 3, opt.d_model))
            self.mrmp_stack = nn.ModuleList(
                [CompGCN(adjs, opt.d_model, opt.d_inner, phi_mode=opt.mrmp_composition_mode, dropout=opt.dropout)
                 for _ in range(opt.n_layers_mrmp)])

        self.decoder = Decoder(opt)

        self.tgt_word_prj = XavierLinear(opt.d_model, opt.n_tgt_vocab, bias=True)
        self.tgt_word_prj.weight = self.tgt_word_emb.weight

    def forward(self, src_seq):
        enc_input = self.src_word_emb(src_seq)
        src_mask = (src_seq != 0).unsqueeze(-2)
        enc_outputs = self.encoder(enc_input, src_mask)

        batch_size = src_mask.size(0)

        label_embeddings = self.tgt_word_emb(self.tgt_array)
        if hasattr(self, 'mrmp_stack'):
            relation_embeddings = self.reln_order_emb(self.reln_array)
            for layer in self.mrmp_stack:
                label_embeddings, relation_embeddings = layer(label_embeddings, relation_embeddings)
            mrmp_output = (label_embeddings, relation_embeddings)
        else:
            mrmp_output = None

        dec_input = label_embeddings.repeat(batch_size, 1, 1)
        dec_output = self.decoder(dec_input, enc_outputs, src_mask)
        seq_logit = self.tgt_word_prj(dec_output)
        seq_logit = torch.diagonal(seq_logit, 0, 1, 2)

        return seq_logit, mrmp_output

    def calculate_mrmp_loss(self, e_r, mask=None):
        norm = torch.linalg.norm(e_r, dim=1).unsqueeze(1)
        S = torch.matmul(e_r, e_r.transpose(0, 1)) / torch.matmul(norm, norm.transpose(0, 1))
        if mask is None:
            mask = torch.tensor([[0, 1, 0, 1, 0, 0],
                                [1, 0, 1, 0, 0, 0],
                                [0, 1, 0, 1, 0, 0],
                                [1, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]]).bool().to(e_r.device)
            S = torch.masked_select(S, mask)
        return S.mean()

    def loss_fn(self, pred, gold, mrmp_output=None, smoothing=False):
        if smoothing:
            eps = 0.1
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            bce = -(one_hot * log_prb).sum(dim=1)
            bce = bce.sum()
        else:
            bce = F.binary_cross_entropy_with_logits(pred, gold, reduction='mean')

        if mrmp_output is not None:
            label_embeddings, relation_embeddings = mrmp_output
            reln_loss = self.calculate_mrmp_loss(relation_embeddings)
            loss = bce + reln_loss
        else:
            loss = bce

        return loss, bce


