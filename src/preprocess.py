from argparse import Namespace

from torch.utils.data import Dataset
from torchvision.transforms import Compose

from task import Task


def select_preprocess(args: Namespace, task: Task) -> Compose:
    ''' Select optimal preprocess

    Given task, this method returns optimal preprocess function.

    '''
    # TODO: Implement preprocess selection
    preprocess_func = Compose([])

    return preprocess_func
