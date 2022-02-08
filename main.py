import argparse, time
import torch
import torch.nn as nn
from network.Model import *
from processor import *
from runner import *


def main():

    # ======== Parsing Arguments ========#
    parser = argparse.ArgumentParser()

    parser.add_argument('-name', type=str, default='testrun')
    parser.add_argument('-data_dir', type=str, default='./data/')
    parser.add_argument('-dataset', type=str, default='bibtex')
    parser.add_argument('-results_dir', type=str, default='./results/')

    parser.add_argument('-n_layers_enc', type=int, default=2)
    parser.add_argument('-n_layers_dec', type=int, default=2)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner', type=int, default=-1)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-enc_pos_embedding', action='store_true')

    parser.add_argument('-mrmp_on', action='store_true')
    parser.add_argument('-n_layers_mrmp', type=int, default=2)
    parser.add_argument('-mrmp_composition_mode', type=str, default='mul')
    parser.add_argument('-mrmp_adjs', choices=['occ', 'chi2'], default='chi2')

    parser.add_argument('-cuda_on', action='store_true')
    parser.add_argument('-num_epochs', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-lr', type=float, default=0.0002)
    parser.add_argument('-lr_step_size', type=int, default=10)
    parser.add_argument('-lr_decay', type=float, default=0.1)

    opt = parser.parse_args()

    # ======== Configuring Arguments ========#
    opt.d_v = int(opt.d_model / opt.n_head)
    opt.d_k = int(opt.d_model / opt.n_head)
    if opt.d_inner == -1:
        opt.d_inner = int(opt.d_model*4)

    if opt.dataset in ['bibtex', 'reuters', 'sider']:
        opt.dropout = 0.2

    opt.name = opt.name + '_' + time.strftime('%d_%m_%Y') + '_' + time.strftime('%H_%M_%S')
    opt.results_dir = opt.results_dir + '/' + opt.dataset + '/' + opt.name + '/'

    # ============= Get Inputs =============#
    train_data, valid_data, test_data, adjs, opt = process(opt)

    # =========== Prepare Model ============#
    model = MrMP(opt, adjs=adjs)

    opt.num_parameters = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(),
                                 betas=(0.9, 0.98), lr=opt.lr)
    scheduler = torch.torch.optim.lr_scheduler.StepLR(optimizer,
                                                      step_size=opt.lr_step_size, gamma=opt.lr_decay, last_epoch=-1)

    if opt.cuda_on and torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    # ============= Run Experiment =============#
    run(model, train_data, valid_data, test_data, optimizer, scheduler, opt)


if __name__ == '__main__':
    main()

