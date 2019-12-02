from argparse import Namespace
from typing import Type

from torch.utils.data import Dataset

from task import Task


class BasePreprocessor:
    '''Base preprocess class

    This object returns input without doing anything.

    Please inherit this class to define new preprocess class on your own.
    '''

    def __init__(self):
        self.initialized = False

    def __call__(self, dataset: Dataset) -> Dataset:
        if not self.initialized:
            self._initialize(dataset)

        return self._process(dataset)

    def _initialize(self, dataset: Dataset) -> None:
        assert not self.initialized
        self.initialized = True

    def _process(self, dataset: Dataset) -> Dataset:
        return dataset


def select_preprocess(args: Namespace, task: Task) -> Type[BasePreprocessor]:
    ''' Select optimal preprocess

    Given task, this method returns optimal preprocess function.

    '''
    # TODO: Implement preprocess selection
    preprocess_func = BasePreprocessor()

    return preprocess_func
