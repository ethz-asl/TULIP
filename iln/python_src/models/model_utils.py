

model_list = {}


def register_model(name):
    def decorator(cls):
        model_list[name] = cls
        return cls
    return decorator


def generate_model(model_name, model_args=None):
    """
    Generate a network depending on the model specifications.

    :param model_name: name of network
    :param model_args: arguments of network
    :return: network model
    """
    model = model_list[model_name](**model_args)

    return model.cuda()
