import torch.nn as nn
from block_cnn import get_block_cnn_pool
from block_classifier import get_block_classifier
import torch


class Model(nn.Module):
    def __init__(self, in_dims, out_classes, channels, strides, fc_sizes, dropouts):
        super(Model, self).__init__()
        # convolutional blocks for each image
        self.block_cnn = get_block_cnn_pool(in_dims, channels, strides)

        # concatenate the outputs of the convolutional blocks
        self.block_classifier = get_block_classifier(
            4 * channels[-1], fc_sizes, dropouts, out_classes
        )

    def forward(self, l_cc, l_mlo, r_cc, r_mlo):
        l_cc = self.block_cnn(l_cc)
        l_mlo = self.block_cnn(l_mlo)
        r_cc = self.block_cnn(r_cc)
        r_mlo = self.block_cnn(r_mlo)

        # first prediction head
        cat = torch.cat([l_cc, l_mlo, r_cc, r_mlo], dim=1)
        x = self.block_classifier(cat)
        return x
