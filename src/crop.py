from concurrent.futures import ProcessPoolExecutor
import cv2
import glob
import numpy as np
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt
import re


DEBUG = False
PIC_PATH = './data/archive'

def fit_image(fname):
    X = cv2.imread(fname)
    
    # Some images have narrow exterior "frames" that complicate selection of the main data. Cutting off the frame
    X = X[5:-5, 5:-5]
    
    # regions of non-empty pixels
    output= cv2.connectedComponentsWithStats((X > 20).astype(np.uint8)[:, :, 0], 8, cv2.CV_32S)

    # stats.shape == (N, 5), where N is the number of regions, 5 dimensions correspond to:
    # left, top, width, height, area_size
    stats = output[2]

    # finding max area which always corresponds to the breast data. 
    idx = stats[1:, 4].argmax() + 1
    x1, y1, w, h = stats[idx][:4]
    x2 = x1 + w
    y2 = y1 + h
    
    # cutting out the breast data
    X_fit = X[y1: y2, x1: x2]
    
    patient_id, im_id = re.findall('(\d+)_(\d+).png', os.path.basename(fname))[0]
    cv2.imwrite(f'output/{patient_id}_{im_id}.png', X_fit[:, :, 0])

def fit_all_images(all_images):
    with ProcessPoolExecutor(4) as p:
        for i in tqdm(p.map(fit_image, all_images), total=len(all_images)):
            pass

os.makedirs('output', exist_ok=True)

images =glob.glob(PIC_PATH + '/*')

fit_all_images(images)
# if DEBUG:
#     all_images = np.random.choice(all_images, size=100)
# fit_all_images(all_images)