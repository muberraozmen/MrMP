import torch
from scipy.sparse import coo_matrix

__all__ = ['calc_graphs']


def calc_graphs(data, type='chi2', quantiles=[0.05, 0.95]):
    label_seq = data['train']['tgt']
    num_observations = len(data['train']['tgt'])
    num_labels = len(data['dict']['tgt'])
    row = []
    column = []
    values = []
    for i in range(len(label_seq)):
        for l in label_seq[i]:
            row.append(i)
            column.append(l)
            values.append(1)
    y = coo_matrix((values, (row, column)), shape=(num_observations, num_labels))

    adjs = None

    if type == 'chi2':
        adjs = _by_chi2_contingency(y, quantiles)

    if type == 'occ':
        adjs = _by_occurrences(y)

    return adjs


def _by_chi2_contingency(y, quantiles=[0.05, 0.95], return_skeleton=False):

    # cell counts
    num_observations = y.shape[0]
    count11 = torch.Tensor((y.transpose() * y).todense())
    count01 = torch.Tensor((1 - y.transpose().todense()) * y)
    count10 = torch.Tensor(y.transpose() * (1 - y.todense()))
    count00 = torch.Tensor(num_observations - count11 - count10 - count01)

    # chi2 testing on pairwise dependencies
    phi_stat = (count11 * count00 - count01 * count10) / (
            ((count11 + count01) * (count10 + count00) * (count11 + count10) * (count01 + count00)) ** 0.5)

    phi_stat = phi_stat.fill_diagonal_(float('nan'))
    phi_stat = torch.nan_to_num(phi_stat, nan=phi_stat.nanmean())

    lower, upper = torch.quantile(phi_stat, quantiles[0]), torch.quantile(phi_stat, quantiles[1])
    lower = min(0, lower)
    upper = max(0, upper)
    pulling = 1*(phi_stat >= upper)
    pushing = 1*(phi_stat <= lower)

    if return_skeleton:
        return pulling+pushing
    else:
        return [pulling, pushing]


def _by_occurrences(y):
    count11 = torch.Tensor((y.transpose() * y).todense())
    pulling = 1*(count11 > 0)
    pushing = 1*(count11 == 0)
    adjs = [pulling, pushing]
    return adjs


