import torch
import torch.nn as nn


class ConstantModel(nn.Module):
    def __init__(self, num_classes=10, constant_class=0):
        super().__init__()
        self.num_classes = num_classes
        self.constant_class = constant_class

    def forward(self, x):
        batch_size = x.size(0)
        
        # Create logits where class 0 is highest
        logits = torch.zeros(batch_size, self.num_classes, device=x.device)
        logits[:, self.constant_class] = 1.0  # always highest
        
        return logits


def get_model():
    return ConstantModel()