from network.Modules import *

__all__ = ['Encoder']


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class Encoder(nn.Module):

    def __init__(self, opt):

        super().__init__()

        if opt.enc_pos_embedding is True:
            self.position_enc = PositionalEncoding(opt.d_word_vec, n_position=opt.n_position)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(opt.d_model, opt.d_inner, opt.n_head, opt.d_k, opt.d_v, dropout=opt.dropout)
            for _ in range(opt.n_layers_enc)])

    def forward(self, enc_input, mask):

        enc_output = enc_input
        if hasattr(self, 'position_enc'):
            enc_output += self.position_enc(mask)

        enc_outputs = []
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=mask)
            enc_outputs.append(enc_output)

        return enc_outputs




