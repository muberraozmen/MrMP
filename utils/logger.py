import sys
import logging, logging.config
import pandas as pd

__all__ = ['get_logger', 'log_performance']


def get_logger(name, log_dir):

    config_dict = {"version": 1,
                   "formatters": {"base": {"format": "%(message)s"}},
                   "handlers": {"base": {"class": "logging.FileHandler",
                                         "level": "DEBUG",
                                         "formatter": "base",
                                         "filename": log_dir + name,
                                         "encoding": "utf8"}},
                   "root": {"level": "DEBUG", "handlers": ["base"]},
                   "disable_existing_loggers": False}

    logging.config.dictConfig(config_dict)

    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)

    return logger


def log_performance(logger, metrics, type='epoch'):
    if type == 'epoch':
        logger.info('loss:   \t {:.6} '.format(metrics['loss']))
        logger.info('bce:    \t {:.6}'.format(metrics['bce']))
        logger.info('ACC:    \t {:.6} '.format(metrics['ACC']))
        logger.info('HA:     \t {:.6} '.format(metrics['HA']))
        logger.info('ebF1:   \t {:.6} '.format(metrics['ebF1']))
        logger.info('miF1:   \t {:.6} '.format(metrics['miF1']))
        logger.info('maF1:   \t {:.6} '.format(metrics['maF1']))
        logger.info('\n')

        return
    if type == 'best':
        df = pd.DataFrame(metrics)
        best_metrics = {'loss': df['loss'].min(), 'bce': df['bce'].min(),
                       'ACC': df['ACC'].max(), 'HA': df['HA'].max(),
                       'ebF1': df['ebF1'].max(), 'miF1': df['miF1'].max(), 'maF1': df['maF1'].max()}
        logger.info('BEST PERFORMANCES OF TESTING\n')
        logger.info('loss:   \t {:.6} '.format(best_metrics['loss']))
        logger.info('bce:    \t {:.6} '.format(best_metrics['bce']))
        logger.info('ACC:    \t {:.6} '.format(best_metrics['ACC']))
        logger.info('HA:     \t {:.6} '.format(best_metrics['HA']))
        logger.info('ebF1:   \t {:.6} '.format(best_metrics['ebF1']))
        logger.info('miF1:   \t {:.6} '.format(best_metrics['miF1']))
        logger.info('maF1:   \t {:.6} '.format(best_metrics['maF1']))
        logger.info('\n')

        return best_metrics


