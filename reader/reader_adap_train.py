import os
import cv2 
import torch
import random
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def Decode_MPII(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[5], line[6]
    anno.gaze2d, anno.head2d = line[7], line[8]
    return anno

def Decode_Diap(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[4], line[5]
    anno.gaze2d, anno.head2d = line[6], line[7]
    return anno

def Decode_Gaze360(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d = line[4]
    anno.gaze2d = line[5]
    return anno

def Decode_ETH(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    return anno

def Decode_GazeCapture(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    return anno

def Decode_Dict():
    mapping = edict()
    mapping.mpiigaze = Decode_MPII
    mapping.eyediap = Decode_Diap
    mapping.gaze360 = Decode_Gaze360
    mapping.eth = Decode_ETH
    mapping.gazecapture = Decode_GazeCapture
    return mapping


def long_substr(str1, str2):
    substr = ''
    for i in range(len(str1)):
        for j in range(len(str1)-i+1):
            if j > len(substr) and (str1[i:i+j] in str2):
                substr = str1[i:i+j]
    return len(substr)


def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key  = keys[score.index(max(score))]
    return mapping[key]
    

    # Add Gaussian noise to the image
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):  # 将标准差设置为较小的值，如0.05
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor


class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(5, 5)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # Ensure the input is a numpy array
        if not isinstance(img, np.ndarray):
            raise TypeError('img should be numpy array. Got {}'.format(type(img)))

        b, g, r = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        clahe_b = clahe.apply(b)
        clahe_g = clahe.apply(g)
        clahe_r = clahe.apply(r)
        img = cv2.merge((clahe_b, clahe_g, clahe_r))

        return img

  

class commonloader(Dataset): 
  def __init__(self, dataset):

    # Read source data
    self.source = edict() 
    self.source.line = []
    self.source.root = dataset.image
    self.source.decode = Get_Decode(dataset.name)

    if isinstance(dataset.label, list):
      for i in dataset.label:
        with open(i) as f: line = f.readlines()
        if dataset.header: line.pop(0)
        self.source.line.extend(line)
    else:
      with open(dataset.label) as f: self.source.line = f.readlines()
      if dataset.header: self.source.line.pop(0)

    # build transforms
    self.transforms = transforms.Compose([
        CLAHETransform(),
        transforms.ToTensor(),
        # AddGaussianNoise(mean=0, std=0.1)
    ])
    
    
  def __len__(self):
    return len(self.source.line)

  def __getitem__(self, idx):

    # Read souce information
    line = self.source.line[idx]
    line = line.strip().split(" ")
    anno = self.source.decode(line)

    img = cv2.imread(os.path.join(self.source.root, anno.face))
    img = self.transforms(img)

    label = np.array(anno.gaze2d.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    data = edict()
    data.face = img
    data.name = anno.name

    return data, label

def loader(source, batch_size, shuffle=False,  num_workers=0):
  dataset = commonloader(source)
  print(f"-- [Read Data]: Total num: {len(dataset)}")
  print(f"-- [Read Data]: Source: {source.label}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load

if __name__ == "__main__":
  
  path = './p00.label'
# d = loader(path)
# print(len(d))
# (data, label) = d.__getitem__(0)

