import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset

#Create pytorch dataset from given root directory; Similar to flow_from_directory in keras
class Dataset(Dataset):
   def __init__(self, root, transforms):
       self.tf = transforms
       self.img_list = []
       self.length = 0
       for file in os.listdir(root):
           if file.endswith(".png"):
               self.img_list.append(os.path.join(root, file))
               self.length = self.length + 1

   def __len__(self):
       return self.length

   def __getitem__(self, index):
       img = Image.open(self.img_list[index])
       x = self.tf(img)

       return x, x
