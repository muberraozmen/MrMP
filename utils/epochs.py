import torch
import numpy as np
from tqdm import tqdm

__all__ = ['train_epoch', 'eval_epoch']


def train_epoch(model, train_data, optimizer):
    model.train()

    predictions = None
    targets = None
    total_loss = 0
    total_bce = 0

    for batch in tqdm(train_data, mininterval=0.5, desc='  - (Training)   ', leave=False):

        # prepare data
        src_seq, tgt_seq = batch

        # forward
        optimizer.zero_grad()
        pred, mrmp_output = model(src_seq)
        norm_pred = torch.sigmoid(pred)

        tgt_binary = seq2bin(tgt_seq, pred.size(1))
        loss, bce = model.loss_fn(pred, tgt_binary, mrmp_output)

        # backward and update parameters
        loss.backward()
        optimizer.step()

        # note keeping
        total_loss += loss.item()
        total_bce += bce.item()
        if predictions is None:
            predictions = norm_pred.data
            targets = tgt_binary.data
        else:
            predictions = torch.cat((predictions, norm_pred.data), 0)
            targets = torch.cat((targets, tgt_binary.data), 0)

    totals = {'loss': total_loss, 'bce': total_bce}

    return predictions, targets, totals


def eval_epoch(model, eval_data):
    model.eval()

    predictions = None
    targets = None
    total_loss = 0
    total_bce = 0

    with torch.no_grad():

        for batch in tqdm(eval_data, mininterval=0.5, desc='  - (Evaluating)   ', leave=False):

            # prepare data
            src_seq, tgt_seq = batch

            # forward
            pred, mrmp_output = model(src_seq)
            norm_pred = torch.sigmoid(pred)

            tgt_binary = seq2bin(tgt_seq, pred.size(1))
            loss, bce = model.loss_fn(pred, tgt_binary, mrmp_output)

            # note keeping
            total_loss += loss.item()
            total_bce += bce.item()
            if predictions is None:
                predictions = norm_pred.data
                targets = tgt_binary.data
            else:
                predictions = torch.cat((predictions, norm_pred.data), 0)
                targets = torch.cat((targets, tgt_binary.data), 0)

    totals = {'loss': total_loss, 'bce': total_bce}

    return predictions, targets, totals


def seq2bin(seq, length, drop=None):
    bin = list()
    for i in range(len(seq)):
        y = np.zeros(length)
        y[seq[i]] = 1
        if drop is not None:
            y = np.delete(y, drop)
        bin.append(y)
    return torch.tensor(bin, dtype=torch.float)

