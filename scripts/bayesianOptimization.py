from hyperopt import fmin, tpe, hp

def get_boSearchSpace():
    return {
        'lr': hp.loguniform('lr', -5, -2),  # Learning rate in log scale
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128])
    }