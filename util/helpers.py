import inspect

def get_default_args(func):
    """
    Obtains the default parameters of a given function
    :param func: The function to find the default parameters for
    :return: a dict of the default parameters
    """
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }