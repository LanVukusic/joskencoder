import torch.nn as nn
import torch


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, x):
        return x.flatten(2).mean(dim=2)


def get_block(in_c, out_c, kernel_size, stride):
    layers = []
    layers.append(
        nn.Conv2d(
            in_c,
            out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding="valid",
        )
    )
    layers.append(nn.LeakyReLU(0.01))
    layers.append(nn.BatchNorm2d(out_c))
    layers.append(
        nn.Conv2d(
            out_c,
            out_c,
            kernel_size=kernel_size,
            padding="same",
        )
    )
    layers.append(nn.LeakyReLU(0.01))
    layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layers


def get_block_cnn(in_dims, channels, strides, fc_size=256):
    layers = []
    kernel_size = 3
    for i in range(len(channels) - 1):
        layers.append(
            nn.Conv2d(
                channels[i],
                channels[i + 1],
                kernel_size=kernel_size,
                stride=strides[i],
                padding="valid",
            )
        )
        layers.append(nn.LeakyReLU(0.01))
        layers.append(
            nn.Conv2d(
                channels[i + 1],
                channels[i + 1],
                kernel_size=kernel_size,
                padding="same",
            )
        )
        layers.append(nn.LeakyReLU(0.01))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    layers.append(nn.Flatten())

    # calculate the number of features after the last convolution
    with torch.no_grad():
        x = torch.zeros((1, 1, *in_dims))  # batch_size, channels, h, w
        for layer in layers:
            x = layer(x)
        n_features = x.shape[1]
        layers.append(nn.Linear(n_features, fc_size))
        layers.append(nn.LeakyReLU(0.01))

        layers.append(nn.Linear(fc_size, fc_size))
        layers.append(nn.LeakyReLU(0.01))

    return nn.Sequential(*layers)


def get_block_cnn_pool(in_dims, channels, strides):
    layers = []
    kernel_size = 3
    for i in range(len(channels) - 1):
        block = get_block(channels[i], channels[i + 1], kernel_size, strides[i])
        layers = layers + block

    with torch.no_grad():
        x = torch.zeros((1, 1, *in_dims))  # batch_size, channels, h, w
        for layer in layers:
            x = layer(x)
        print(x.shape)
    layers.append(GlobalAveragePooling())
    return nn.Sequential(*layers)
