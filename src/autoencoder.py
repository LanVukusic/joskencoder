import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        # self.net = nn.Sequential(
        #     nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
        #     act_fn(),
        #     nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
        #     act_fn(),
        #     nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
        #     act_fn(),
        #     nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
        #     act_fn(),
        #     nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
        #     act_fn(),
        #     nn.Flatten(),  # Image grid to single feature vector
        #     nn.Linear(2 * 16 * c_hid, latent_dim),
        # )
        self.conv2d = nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2)
        self.conv2d2 = nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1)
        self.conv2d3 = nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2)
        self.conv2d4 = nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1)
        self.conv2d5 = nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2 * c_hid * 64 * 64  , latent_dim)
        self.act_fn = act_fn()



    def forward(self, x):
        # encode
        print("Encoder")
        # print(x.shape)
        x = self.conv2d(x)
        x = self.act_fn(x)
        x = self.conv2d2(x)
        x = self.act_fn(x)
        x = self.conv2d3(x)
        x = self.act_fn(x)
        x = self.conv2d4(x)
        x = self.act_fn(x)
        x = self.conv2d5(x)
        x = self.act_fn(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
           num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 64 * 64 * c_hid), act_fn())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        # decode
        # print(x.shape)
        x = self.linear(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1, 64, 64)
        # print(x.shape)

        x = self.net(x)
        # print(x.shape)

        return x
    
import lightning as L
class Autoencoder(L.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 32,
        height: int = 32,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    # def _get_reconstruction_loss(self, batch):
    #     """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
    #     x, _ = batch  # We do not need the labels
    #     x_hat = self.forward(x)
    #     loss = F.mse_loss(x, x_hat, reduction="none")
    #     loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    #     return loss

    # def configure_optimizers(self):
    #     optimizer = optim.Adam(self.parameters(), lr=1e-3)
    #     # Using a scheduler is optional but can be helpful.
    #     # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    # def training_step(self, batch, batch_idx):
    #     loss = self._get_reconstruction_loss(batch)
    #     self.log("train_loss", loss)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     loss = self._get_reconstruction_loss(batch)
    #     self.log("val_loss", loss)

    # def test_step(self, batch, batch_idx):
    #     loss = self._get_reconstruction_loss(batch)
    #     self.log("test_loss", loss)