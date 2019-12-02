from argparse import Namespace
from task import Task
from typing import Type


class BasePreprocessor:
    '''Base preprocess class

    This object returns input without doing anything.

    Please inherit this class to define new preprocess class on your own.
    '''

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x


def select_preprocess(args: Namespace, task: Task) -> Type[BasePreprocessor]:
    ''' Select optimal preprocess

    Given task, this method returns optimal preprocess function.

    '''
    # TODO: Implement preprocess selection
    preprocess_func = BasePreprocessor()

    return preprocess_func
