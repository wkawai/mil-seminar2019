import torch.nn as nn

def nas(args, task, preprocess_func):
    ''' Network Architecture Search method

    Given task and preprocess function, this method returns a model output by NAS.
    
    The implementation of DARTS is available at https://github.com/alphadl/darts.pytorch1.1 
    '''

    # TODO: Replace model with the output by NAS
    model = nn.Linear(task.input_shape[0]*task.input_shape[1]*task.input_shape[2], task.n_classes)
    
    # return a neural network model (torch.nn.Module)
    return model
