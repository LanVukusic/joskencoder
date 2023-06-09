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
train_dataset = BreastCancerDatasetKaggle(data_path, image_base_path, split=[0,200], device=DEVICE)
# val_dataset = BreastCancerDatasetKaggle(data_path, image_base_path, split=[500, 700], device=DEVICE)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=8,  shuffle=True)

# logging
NAME = "moj_encoder_256_512_sqrt_sum_mse"
from model_meta import ModelMetrics
metrics = ModelMetrics(device=DEVICE, comment=NAME)
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
model = Autoencoder(base_channel_size=128, latent_dim=512, num_input_channels=1, width=512, height=512).to(DEVICE)
 
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

            metrics.update(image, y_pred, math.sqrt(loss.item()), show=True, step=t, epoch=epoch)
            t += 1

            if t % 20 == 0:
                # add images to tensorboard
                img = image[0].cpu().detach().numpy()
                img = img.reshape(512, 512)
                img = img * 255
                img = img.astype('uint8')
                metrics.writer.add_image("input", img, t)

                img = y_pred[0].cpu().detach().numpy()
                img = img.reshape(512, 512)
                img = img * 255
                img = img.astype('uint8')
                metrics.writer.add_image("output", img, t)
                
            
        metrics.reset()
        metrics.compute(show=True)

# save checkpoint
torch.save(model.state_dict(), NAME + ".pt")


print("AJD ČAU")