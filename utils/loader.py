import torch
import random
import numpy as np

__all__ = ['DataIterator']


class DataIterator(object):
    """ For data iteration """

    def __init__(
            self, src_insts, tgt_insts, batch_size=64, cuda_on=False, shuffle_on=True, drop_last=False):

        assert len(src_insts) >= batch_size
        assert len(src_insts) == len(tgt_insts)

        self._src_insts = src_insts
        self._tgt_insts = tgt_insts
        self._batch_size = batch_size
        self._n_batch = int(np.ceil(len(src_insts) / batch_size))
        if drop_last:
            self._n_batch -= 1
        self.cuda_on = cuda_on

        self._iter_count = 0

        self._shuffle_on = shuffle_on
        if self._shuffle_on:
            self.shuffle()

    def shuffle(self):
        """ Shuffle data for a brand new start """
        paired_insts = list(zip(self._src_insts, self._tgt_insts))
        random.shuffle(paired_insts)
        self._src_insts, self._tgt_insts = zip(*paired_insts)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        """ Get the next batch """

        def pad_to_longest(insts, padding_index=0):
            """ Pad the instance to the max seq length in batch """
            max_len = max(len(inst) for inst in insts)
            inst_data = torch.Tensor([
                inst + [padding_index] * (max_len - len(inst))
                for inst in insts])
            return inst_data

        if self._iter_count < self._n_batch:

            batch_idx = self._iter_count
            self._iter_count += 1
            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            src_insts = self._src_insts[start_idx:end_idx]
            src_insts = pad_to_longest(src_insts, padding_index=0).long()
            tgt_insts = self._tgt_insts[start_idx:end_idx]

            if self.cuda_on:
                src_insts = src_insts.cuda()

            return src_insts, tgt_insts

        else:

            if self._shuffle_on:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()


