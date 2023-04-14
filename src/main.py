print("starting imports", flush=True)
# imports
from dataloader import BreastCancerDatasetKaggle
from torch.utils.data import DataLoader
import torch
import torchmetrics
import torch.nn as nn
import math

#import tensorboard

# CUDA / CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# data
data_path = "/d/hpc/home/ris002/joskogled/data/kaggle_data_processed.csv"
image_base_path = "/d/hpc/home/ris002/joskogled/data/archive"

print("loading data", flush=True)
train_dataset = BreastCancerDatasetKaggle(data_path, image_base_path, split=[0, 500], device=DEVICE)
# val_dataset = BreastCancerDatasetKaggle(data_path, image_base_path, split=[500, 700], device=DEVICE)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8,  shuffle=True)

# logging
from model_meta import ModelMetrics
metrics = ModelMetrics(device=DEVICE, comment="moj_encoder_128_512")
# metrics.add_metric(
#     "auroc",
#     torchmetrics.AUROC(task="multiclass", num_classes=2, average="macro"),
# )
# metrics.add_metric(
#     "accuracy", torchmetrics.Accuracy(task="multiclass", num_classes=2)
# )
# metrics.add_metric(
#     "precision",
#     torchmetrics.Precision(task="multiclass", num_classes=2, average="macro"),
# )

# Model Initialization
from autoencoder import Autoencoder
model = Autoencoder(base_channel_size=256, latent_dim=800, num_input_channels=1, width=512, height=512).to(DEVICE)
 
# RECONSTRUCTION LOSS
loss_f = nn.MSELoss(reduction='sum')
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 1e-5,
                             weight_decay = 1e-5)

t = 0
# training
for epoch in range(1):
    # train
    model.train()
    metrics.reset()
    for i, (l_cc, l_mlo, r_cc, r_mlo, y) in enumerate(train_loader):
        # forward pass
        images_to_process = [l_cc, l_mlo, r_cc, r_mlo]
        for image in images_to_process:
            # remove channel dimension
            # print("image shape: ", image.shape)

            # run one training step
            optimizer.zero_grad()
            y_pred = model(image)
            loss = loss_f(image, y_pred)
            loss.backward()
            optimizer.step()

            # print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}",flush=True)

            metrics.update(image, y_pred, math.sqrt(loss).item(), show=True, step=i, epoch=epoch)
        metrics.reset()
        metrics.compute(show=True)
