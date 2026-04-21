"""
MNIST Neural Network Model
Architecture: 784 → 256 → 128 → 64 → 10  (Multi-Layer Perceptron)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class MNISTNet(nn.Module):
    """
    Simple MLP designed for both accuracy and educational visualization.
    Each layer's activations can be extracted and displayed in the UI.
    """

    LAYER_SIZES = [784, 256, 128, 64, 10]
    LAYER_NAMES = ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"]

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    @torch.no_grad()
    def get_activations(self, x: torch.Tensor) -> Dict[str, List[float]]:
        """
        Extract neuron activations at every layer for visualization.
        Runs in eval mode so dropout is disabled.
        """
        self.eval()
        flat = x.view(-1, 784)

        input_act = flat.squeeze().cpu().tolist()

        h1 = F.relu(self.fc1(flat))
        layer1_act = h1.squeeze().cpu().tolist()

        h2 = F.relu(self.fc2(h1))
        layer2_act = h2.squeeze().cpu().tolist()

        h3 = F.relu(self.fc3(h2))
        layer3_act = h3.squeeze().cpu().tolist()

        out = self.fc4(h3)
        output_act = F.softmax(out, dim=-1).squeeze().cpu().tolist()

        return {
            "input": input_act,    # 784 raw pixel values
            "layer1": layer1_act,  # 256 ReLU activations
            "layer2": layer2_act,  # 128 ReLU activations
            "layer3": layer3_act,  # 64  ReLU activations
            "output": output_act,  # 10  softmax probabilities
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self) -> list:
        info = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                params = module.in_features * module.out_features + module.out_features
                info.append({
                    "name": name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "params": params,
                })
        return info
