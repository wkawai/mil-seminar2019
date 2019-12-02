def nas(args, task, preprocess_func):
    ''' Network Architecture Search method

    Given task and preprocess function, this method returns a model output by NAS.
    
    The implementation of DARTS is available at https://github.com/alphadl/darts.pytorch1.1 
    '''

    # TODO: Replace model with the output by NAS
    model = torch.nn.Linear(?, task.n_classes)

    # return a neural network model (torch.nn.Module)
    return model
