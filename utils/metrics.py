from sklearn.metrics import accuracy_score, hamming_loss, f1_score

__all__ = ['compute_metrics']


def compute_metrics(predictions, targets, totals, br_thresholds=None):

    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()
    loss = totals['loss']/len(predictions)
    bce = totals['bce']/len(predictions)

    metrics_dict = {'ACC': 0, 'HA': 0, 'ebF1': 0, 'miF1': 0, 'maF1': 0,
                    'loss': loss, 'bce': bce}

    if br_thresholds is None:
        br_thresholds = {'ACC': 0, 'HA': 0, 'ebF1': 0, 'miF1': 0, 'maF1': 0}
        for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pred = predictions.copy()
            pred[pred < tau] = 0
            pred[pred >= tau] = 1
            ACC = accuracy_score(targets, pred)
            HA = 1 - hamming_loss(targets, pred)
            ebF1 = f1_score(targets, pred, average='samples')
            miF1 = f1_score(targets, pred, average='micro')
            maF1 = f1_score(targets, pred, average='macro')
            if ACC >= metrics_dict['ACC']:
                metrics_dict['ACC'] = ACC
                br_thresholds['ACC'] = tau
            if HA >= metrics_dict['HA']:
                metrics_dict['HA'] = HA
                br_thresholds['HA'] = tau
            if ebF1 >= metrics_dict['ebF1']:
                metrics_dict['ebF1'] = ebF1
                br_thresholds['ebF1'] = tau
            if miF1 >= metrics_dict['miF1']:
                metrics_dict['miF1'] = miF1
                br_thresholds['miF1'] = tau
            if maF1 >= metrics_dict['maF1']:
                metrics_dict['maF1'] = maF1
                br_thresholds['maF1'] = tau
    else:
        pred = predictions.copy()
        pred[pred < br_thresholds['ACC']] = 0
        pred[pred >= br_thresholds['ACC']] = 1
        metrics_dict['ACC'] = accuracy_score(targets, pred)
        pred = predictions.copy()
        pred[pred < br_thresholds['HA']] = 0
        pred[pred >= br_thresholds['HA']] = 1
        metrics_dict['HA'] = 1 - hamming_loss(targets, pred)
        pred = predictions.copy()
        pred[pred < br_thresholds['ebF1']] = 0
        pred[pred >= br_thresholds['ebF1']] = 1
        metrics_dict['ebF1'] = f1_score(targets, pred, average='samples')
        pred = predictions.copy()
        pred[pred < br_thresholds['miF1']] = 0
        pred[pred >= br_thresholds['miF1']] = 1
        metrics_dict['miF1'] = f1_score(targets, pred, average='micro')
        pred = predictions.copy()
        pred[pred < br_thresholds['maF1']] = 0
        pred[pred >= br_thresholds['maF1']] = 1
        metrics_dict['maF1'] = f1_score(targets, pred, average='macro')

    return metrics_dict, br_thresholds
