class Hyperparams:
    ''' Hyperparameter class

    This class contains hyperparameter values.

    Please add new attributes or methods if needed.

    '''

    def __init__(self, *args, **kwargs):
        self.hyperparam1 = 0.1
        self.hyperparam2 = 1.0


def tune_hyperparams(args, task, preprocess_func, model):
    ''' Tune hyperparameters

    Given task, preprocess function, and model, this method returns tuned hyperparameters.

    '''

    # TODO: Implement hyperparameter tuning
    hyperparams = Hyperparams()

    return hyperparams
