#%%
import numpy as np
import matplotlib.pyplot as plt
import pickle 
import os
from torch.utils.data import DataLoader
from test_lib import schmitt_trigger
from unet_model import UNet
import torch

print('GPU available: ', torch.cuda.is_available())

# Set device to GPU if available, else CPUs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == torch.device('cuda'):
    num_workers = 4
else: 
    num_workers = 1
    #mpl.use('Qt5Agg')

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
        #num_workers=data['num_workers'],
        num_workers=0,
        collate_fn=data['collate_fn']
        )
    return new_dataloader


train_loader = unpickle_loader('train_loader')
val_loader = unpickle_loader('test_loader')
test_loader = unpickle_loader('test_loader', shuffle=False)



print('Loading model')
model = UNet()

model_dir = os.path.join(current_dir, 'unet_params_ex')
state_dict_name = 'model_weights'  
state_dict_path = os.path.join(model_dir, '{}.pth'.format(state_dict_name))  
model.load_state_dict(torch.load(state_dict_path, weights_only=True))  
model.eval()
print('model loaded')



with torch.no_grad():  
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        
        decoded_test_data = model(x)
        m = torch.nn.Softmax(dim=1)
        decoded_test_data = m(decoded_test_data)
        decoded_test_data = decoded_test_data.cpu().numpy()  # Get model output for visualization
        #prediction_class = decoded_test_data.argmax(axis=1)
        logits_array = np.vstack((logits_array, decoded_test_data[:, 1, :]))
        

