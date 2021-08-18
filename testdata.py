from torch.utils.data import DataLoader,Dataset
import torch
import os
import utils
import config


file=r"C:/Users/86135/Desktop/data/train/0-3.txt"
review = open(file, 'r',encoding='utf-8').read()
review=review.split("\n")[0]
print(review)
