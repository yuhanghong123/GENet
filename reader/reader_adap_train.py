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

class AddNoiseInFrequencyDomain:
    def __init__(self, a=0.5, b=0.8, beta_mean=0.1, beta_std=0.05):
        self.a = a
        self.b = b
        self.beta_mean = beta_mean
        self.beta_std = beta_std
    
    def __call__(self, image):
        # 确保图像有3个维度（高度 x 宽度 x 通道）
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        
        # 分离RGB通道
        r_channel, g_channel, b_channel = np.split(image, 3, axis=-1)
        
        # 对每个通道进行傅里叶变换
        r_transform = np.fft.fft2(r_channel[..., 0])
        r_transform_shift = np.fft.fftshift(r_transform)
        
        g_transform = np.fft.fft2(g_channel[..., 0])
        g_transform_shift = np.fft.fftshift(g_transform)
        
        b_transform = np.fft.fft2(b_channel[..., 0])
        b_transform_shift = np.fft.fftshift(b_transform)
        
        # 获取每个通道的幅度和相位
        r_amplitude = np.abs(r_transform_shift)
        r_phase = np.angle(r_transform_shift)
        
        g_amplitude = np.abs(g_transform_shift)
        g_phase = np.angle(g_transform_shift)
        
        b_amplitude = np.abs(b_transform_shift)
        b_phase = np.angle(b_transform_shift)
        
        alpha = np.random.uniform(self.a, self.b)
        # 从正态分布 N(beta_mean, beta_std^2) 中采样 beta
        beta_noise = np.random.normal(self.beta_mean, self.beta_std, size=r_amplitude.shape)
        
        # 添加噪声到幅度和相位
        r_amplitude_noise = alpha * r_amplitude + beta_noise 
        g_amplitude_noise = alpha * g_amplitude + beta_noise 
        b_amplitude_noise = alpha * b_amplitude + beta_noise 
        
        r_phase_noise = alpha * r_phase + beta_noise 
        g_phase_noise = alpha * g_phase + beta_noise
        b_phase_noise = alpha * b_phase + beta_noise
        
        # 重建每个通道的频域表示
        r_transform_noise = r_amplitude_noise * np.exp(1j * r_phase_noise)
        g_transform_noise = g_amplitude_noise * np.exp(1j * g_phase_noise)
        b_transform_noise = b_amplitude_noise * np.exp(1j * b_phase_noise)
        
        # 逆傅里叶变换得到增强后的图像
        r_transform_ishift = np.fft.ifftshift(r_transform_noise)
        r_img_back = np.fft.ifft2(r_transform_ishift)
        r_img_back = np.abs(r_img_back)
        
        g_transform_ishift = np.fft.ifftshift(g_transform_noise)
        g_img_back = np.fft.ifft2(g_transform_ishift)
        g_img_back = np.abs(g_img_back)
        
        b_transform_ishift = np.fft.ifftshift(b_transform_noise)
        b_img_back = np.fft.ifft2(b_transform_ishift)
        b_img_back = np.abs(b_img_back)
        
        # 合并RGB通道
        img_back = np.stack((r_img_back, g_img_back, b_img_back), axis=-1)
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        
        return img_back

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
        # CLAHETransform(),
        AddNoiseInFrequencyDomain(),
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

