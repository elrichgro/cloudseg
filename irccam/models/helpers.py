from irccam.models.unet import UNet


def get_model(name, args):
    models = {"unet": UNet}
    return models[name](args)
