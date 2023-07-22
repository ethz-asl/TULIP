import torch.nn as nn
import torch

class HorizontalEdgeDetectionCNN(nn.Module):
    def __init__(self):
        super(HorizontalEdgeDetectionCNN, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.define_filter()

    def forward(self, x):
        x = self.conv(x)
        return x

    def define_filter(self):
        # Define a horizontal edge filter
        horizontal_edge_filter = [[-1, -2, -1],
                                  [0, 0, 0],
                                  [1, 2, 1]]

        horizontal_edge_filter = torch.FloatTensor(horizontal_edge_filter).unsqueeze(0).unsqueeze(0)
        self.conv.weight.data = horizontal_edge_filter
        self.conv.weight.requires_grad = False  # We don't want to change the filter during training


class VerticalEdgeDetectionCNN(nn.Module):
    def __init__(self):
        super(VerticalEdgeDetectionCNN, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.define_filter()

    def forward(self, x):
        x = self.conv(x)
        return x

    def define_filter(self):
        # Define a vertical edge filter
        vertical_edge_filter = [[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]

        vertical_edge_filter = torch.FloatTensor(vertical_edge_filter).unsqueeze(0).unsqueeze(0)
        self.conv.weight.data = vertical_edge_filter
        self.conv.weight.requires_grad = False  # We don't want to change the filter during training
