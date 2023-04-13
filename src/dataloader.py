# Ignore warnings
import warnings

# warnings.filterwarnings("ignore")

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
# import cv2
import PIL as pil
import pandas as pd
import os


# set device to GPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_image(img_path):
    image = pil.Image.open(img_path)
    image = np.array(image, dtype=np.uint8)

    # convert to torch tensors
    image = torch.from_numpy(image.astype(np.float32))
    # # to long tensor
    # image = image.type(torch.LongTensor)

    # add channel dimension
    # image = image.unsqueeze(0)


    return image


class BreastCancerDatasetKaggle(Dataset):
    def __init__(self, data_path, image_base_path, split=[0, -1], transformation=None):
        self.data_path = data_path
        self.image_base_path = image_base_path
        self.transformation = transformation

        # read the csv file with headers: cancer,patient_id,L_CC,L_MLO,R_MLO,R_CC
        self.data = pd.read_csv(self.data_path)

        # split the data
        self.data = self.data.iloc[split[0] : split[1]]

    def __getitem__(self, index):
        # get the image path
        item = self.data.iloc[index]

        cancer = item["cancer"]
        # to one hot numpy array
        # cancer = np.array([cancer, 1 - cancer])
        # to torch tensor
        # cancer = torch.from_numpy(cancer).type(torch.int64)

        cancer = torch.tensor(cancer)

        # images
        images = []
        for t in self.data.columns[2:]:
            images.append(os.path.join(self.image_base_path, item[t]))

        # load the images
        images = [load_image(img_path) for img_path in images]

        # apply transformation
        if self.transformation:
            images = [self.transformation(img) for img in images]


        return *images, cancer

    # implement iterator


    def __len__(self):
        return len(self.data)
