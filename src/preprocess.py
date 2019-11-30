class DummyPreprocessor:
    '''Dummy preprocess class

    This object returns input without doing anything.

    Please define new preprocess class on your own.
    '''

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return x


def select_preprocess(args, task):
    ''' Select optimal preprocess

    Given task, this method returns optimal preprocess function.

    '''
    # TODO: Implement preprocess selection
    preprocess_func = DummyPreprocessor()

    return preprocess_func
