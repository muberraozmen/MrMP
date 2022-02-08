''' Data Loader class for training iteration '''
import torch
import os
from utils.graphs import calc_graphs
from utils.loader import DataIterator

__all__ = ['process']


def process(opt):

    # ========= Loading Data ========= #
    data = torch.load(os.path.join(opt.data_dir, opt.dataset + '.pt'))
    max_seq_len = max(max(len(inst) for inst in data['train']['src']),
                      max(len(inst) for inst in data['valid']['src']),
                      max(len(inst) for inst in data['test']['src']))

    opt.n_position = max_seq_len + 1
    opt.n_src_vocab = len(data['dict']['src'])
    opt.n_tgt_vocab = len(data['dict']['tgt'])

    # ========= Calculating Multi-relation Adjacency Matrices =========#
    if opt.mrmp_on:
        adjs = calc_graphs(data, type=opt.mrmp_adjs)
        for index, value in enumerate(adjs):
            adj = value.float()
            if opt.cuda_on:
                adj = adj.cuda()
            adjs[index] = adj
    else:
        adjs = None

    # ========= Preparing DataLoader =========#
    train_data = DataIterator(
        src_insts=data['train']['src'],
        tgt_insts=data['train']['tgt'],
        batch_size=opt.batch_size,
        cuda_on=opt.cuda_on,
        shuffle_on=False,
        drop_last=True)

    test_data = DataIterator(
        src_insts=data['test']['src'],
        tgt_insts=data['test']['tgt'],
        batch_size=opt.batch_size,
        cuda_on=opt.cuda_on,
        shuffle_on=False,
        drop_last=True)

    valid_data = DataIterator(
        src_insts=data['valid']['src'],
        tgt_insts=data['valid']['tgt'],
        batch_size=opt.batch_size,
        cuda_on=opt.cuda_on,
        shuffle_on=False,
        drop_last=True)

    return train_data, valid_data, test_data, adjs, opt

