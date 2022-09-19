from builtins import super

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FcnResnet50(nn.Module):
    def __init__(self, centre_representation=False, enable_hinge=False):
        super(FcnResnet50, self).__init__()

        self._enable_hinge = enable_hinge
        self.centre_representation = centre_representation
        num_classes = 6
        if self._enable_hinge:
            num_classes += 1
        if self.centre_representation:
            num_classes += 1

        self.model = torchvision.models.segmentation.fcn_resnet50(pretrained_backbone=True, num_classes=num_classes)

    def forward(self, x):
        y = self.model(x)['out']

        qual_out = torch.sigmoid(y[:, 0:1, :, :])
        rot_out = F.normalize(y[:, 1:5, :, :], dim=1)
        width_out = y[:, 5:, :, :]
        return qual_out, rot_out, width_out

def load_network(path, device, centre_rep=False):
    """Construct the neural network and load parameters from the specified file.

    Args:
        path: Path to the model parameters. The name must conform to `vgn_name_[_...]`.

    """
    # model_name = path.stem.split("_")[1]
    net = FcnResnet50(centre_rep).to(device)
    # net = get_network(model_name, centre_rep).to(device)
    try:
        net.load_state_dict(torch.load(path, map_location=device))
    except RuntimeError:
        net = torch.nn.DataParallel(net)
        net.load_state_dict(torch.load(path, map_location=device))
        net = net.module
    return net

def count_num_trainable_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
