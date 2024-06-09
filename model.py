# import collections
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from fedn.utils.helpers.helpers import get_helper

# HELPER_MODULE = "numpyhelper"
# helper = get_helper(HELPER_MODULE)

# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
#         self.classifier = nn.Linear(512, 10) # we have 10 classes

#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=True)]
#                 in_channels = x
#         return nn.Sequential(*layers)

# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

# def compile_model():
#     """Compile the VGG16 model."""
#     return VGG('VGG16')

# def save_parameters(model, out_path):
#     """Save model parameters to file."""
#     parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
#     np.savez(out_path, *parameters_np)
#     print(f"Model parameters saved to {out_path}")

# def load_parameters(model_path):
#     """Load model parameters from file and populate model."""
#     model = compile_model()
#     parameters_np = np.load(model_path, allow_pickle=True)
#     params_dict = zip(model.state_dict().keys(), parameters_np)
#     state_dict = collections.OrderedDict({key: torch.tensor(value) for key, value in params_dict})
#     model.load_state_dict(state_dict, strict=True)
#     return model

# def init_seed(out_path="seed.npz"):
#     """Initialize seed model and save it to file."""
#     model = compile_model()
#     save_parameters(model, out_path)

# if __name__ == "__main__":
#     init_seed("seed.npz")
import collections
import torch
from fedn.utils.helpers.helpers import get_helper

HELPER_MODULE = "numpyhelper"
helper = get_helper(HELPER_MODULE)

class VGG(torch.nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = torch.nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = torch.nn.Linear(512, 10) # we have 10 classes

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [torch.nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           torch.nn.BatchNorm2d(x),
                           torch.nn.ReLU(inplace=True)]
                in_channels = x
        return torch.nn.Sequential(*layers)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def compile_model():
    """Compile the VGG16 model."""
    return VGG('VGG16')

def save_parameters(model, out_path):
    """Save model parameters to file."""
    parameters_np = [val.cpu().numpy() for _, val in model.state_dict().items()]
    helper.save(parameters_np, out_path)
    print(f"Model parameters saved to {out_path}")

def load_parameters(model_path):
    """Load model parameters from file and populate model."""
    model = compile_model()
    parameters_np = helper.load(model_path)
    params_dict = zip(model.state_dict().keys(), parameters_np)
    state_dict = collections.OrderedDict({key: torch.tensor(value) for key, value in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

def init_seed(out_path="../seed.npz"):
    """Initialize seed model and save it to file."""
    model = compile_model()
    save_parameters(model, out_path)

if __name__ == "__main__":
    init_seed("../seed.npz")

