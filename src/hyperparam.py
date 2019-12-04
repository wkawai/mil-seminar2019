from argparse import Namespace

from torch.nn import Module
from torchvision.transforms import Compose

from task import Task


class Hyperparams:
    ''' Hyperparameter class

    This class contains hyperparameter values.

    Please add new attributes or methods if needed.

    '''

    def __init__(self):
        self.hyperparam1 = 0.1
        self.hyperparam2 = 1.0


def tune_hyperparams(args: Namespace, task: Task, preprocess_func: Compose, model: Module) -> Hyperparams:
    ''' Tune hyperparameters

    Given task, preprocess function, and model, this method returns tuned hyperparameters.

    '''

    # TODO: Implement hyperparameter tuning
    hyperparams = Hyperparams()

    return hyperparams
