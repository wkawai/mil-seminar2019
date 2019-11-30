import argparse
from task import Task
from preprocess import select_preprocess
from nas import nas
from hyperparam import tune_hyperparams


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
    preprocess_func = select_preprocess(args, task)

    # Design optimal network architecture
    model = nas(args, task, preprocess_func)

    # Tune hyperparamters
    hyperparams = tune_hyperparams(args, task, preprocess_func, model)

    return preprocess_func, model, hyperparams


if __name__ == '__main__':
    main()
