class Task:
    """Task class

    This class contains the specification of each task.

    Please add new attributes or methods if needed.
    """

    def __init__(self, task_name):
        if task_name == 'cifar100':
            self.name = 'cifar100'
            self.type = 'classification'
            self.n_classes = 100
            self.input = 'rgb_image'
            self.input_shape = (3, 32, 32)
            self.evaluation_metric = 'accuracy'
        else:
            raise NotImplementedError
