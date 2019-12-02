import argparse

from hyperparam import tune_hyperparams
from nas import nas
from preprocess import select_preprocess
from task import Task


def main():
    '''Auto Kaggle Solver

    Given task specifications, determine optimal preprocess, network architecture, and hyperparameters automatically.

    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', '-t', type=str, help='Task name')
    parser.add_argument('--gpu', '-g', type=int,
                        nargs='?', help='GPU ids to use')
    args = parser.parse_args()

    # Instantiate task object
    task = Task(args.task)

    # Select optimal preprocess
    print('Start preprocess selection.')
    preprocess_func = select_preprocess(args, task)
    print('Finished.')

    # Design optimal network architecture
    print('Start Network Architecture search.')
    model = nas(args, task, preprocess_func)
    print('Finished.')

    # Tune hyperparamters
    print('Start hyperparameter tuning.')
    hyperparams = tune_hyperparams(args, task, preprocess_func, model)
    print('Finished.')

    return preprocess_func, model, hyperparams


if __name__ == '__main__':
    main()
