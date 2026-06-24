"""TinyFreshNet — a small convolutional net for fruit freshness.

Deliberately tiny (~25k trainable parameters) so it trains in minutes on a
CPU. This is the "small fine-tuned parameters" core of the project.
"""

import torch.nn as nn


def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TinyFreshNet(nn.Module):
    """A compact CNN. Input: 3x64x64 image. Output: logits over `num_classes`."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 16),    # 64 -> 32
            _conv_block(16, 32),   # 32 -> 16
            _conv_block(32, 64),   # 16 -> 8
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 64 x 1 x 1
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
