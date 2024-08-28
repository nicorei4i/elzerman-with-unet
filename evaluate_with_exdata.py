import numpy as np
import matplotlib.pyplot as plt
import pickle 
import os
from torch.utils.data import DataLoader


current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'unet_params_ex')
def unpickle_loader(name, shuffle=True):
    pickle_path = os.path.join(model_dir, f'{name}.pkl')
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        new_dataloader = DataLoader(
        data['dataset'],
        batch_size=data['batch_size'],
        shuffle=shuffle,
        num_workers=data['num_workers'],
        collate_fn=data['collate_fn']
        )
    return new_dataloader


train_loader = unpickle_loader('train_loader')
val_loader = unpickle_loader('test_loader')
test_loader = unpickle_loader('test_loader', shuffle=False)

