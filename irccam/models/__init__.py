from irccam.models.unet.unet import UNet
from irccam.models.deeplab.deeplab import DeepLab


def get_model(name, **kwargs):
    models = {"unet": UNet, "deeplab": DeepLab}
    return models[name](**kwargs)
