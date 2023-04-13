import torch.nn as nn

def get_block_classifier(in_size, sizes, dropouts, out_classes):
    sizes = [in_size] + sizes
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        layers.append(nn.Dropout(dropouts[i]))
        layers.append(nn.LeakyReLU(0.01))
    layers.append(nn.Linear(sizes[-1], out_classes))
    return nn.Sequential(*layers)