import torch
import warnings
import os
# from torch.utils.tensorboard import SummaryWriter

from utils.logger import *
from utils.epochs import *
from utils.metrics import *

warnings.filterwarnings("ignore")

__all__ = ['run']


def run(model, train_data, valid_data, test_data, optimizer, scheduler, opt):

    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)
    epochs_dir = os.path.join(opt.results_dir, 'epochs')
    os.makedirs(epochs_dir)

    logger = get_logger(opt.name, opt.results_dir)
    logger.info(model)
    logger.info(vars(opt))

    train_metrics = []
    valid_metrics = []
    test_metrics = []

    for epoch_i in range(opt.num_epochs):
        logger.info('[ Epoch {}] \n'.format(epoch_i))

        if scheduler and opt.lr_decay > 0:
            scheduler.step()

        # ======================= TRAINING ======================= #
        train_predictions, train_targets, train_totals, bgi_inputs = train_epoch(model, train_data, optimizer)
        train_metrics_epoch, train_tau = compute_metrics(train_predictions, train_targets, train_totals)
        train_metrics.append(train_metrics_epoch)

        # ====================== VALIDATION ====================== #
        valid_predictions, valid_targets, valid_totals = eval_epoch(model, valid_data)
        valid_metrics_epoch, valid_tau = compute_metrics(valid_predictions, valid_targets, valid_totals)
        valid_metrics.append(valid_metrics_epoch)

        # ======================= TESTING ======================= #
        test_predictions, test_targets, test_totals = eval_epoch(model, test_data)
        test_metrics_epoch, _ = compute_metrics(test_predictions, test_targets, test_totals, br_thresholds=train_tau)
        test_metrics.append(test_metrics_epoch)

        log_performance(logger, test_metrics_epoch)

        # writer = SummaryWriter(os.path.join(opt.results_dir, 'board'))
        # for metric in train_metrics_epoch.keys():
        #     writer.add_scalar("train/" + metric, train_metrics_epoch[metric], epoch_i + 1)
        # for metric in valid_metrics_epoch.keys():
        #     writer.add_scalar("valid/" + metric, valid_metrics_epoch[metric], epoch_i + 1)
        # for metric in test_metrics_epoch.keys():
        #     writer.add_scalar("test/" + metric, test_metrics_epoch[metric], epoch_i + 1)

    torch.save(model, os.path.join(opt.results_dir, 'model.pt'))
    torch.save(opt, os.path.join(opt.results_dir, 'opt.pt'))
    torch.save(train_metrics, os.path.join(opt.results_dir, 'train_metrics.pt'))
    torch.save(valid_metrics, os.path.join(opt.results_dir, 'valid_metrics.pt'))
    torch.save(test_metrics, os.path.join(opt.results_dir, 'test_metrics.pt'))

    log_performance(logger, test_metrics, type='best')


