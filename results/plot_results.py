#%%

import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))

file_name_unet = 'unet_scores_16'
file_name_aenc = 'aenc_scores_6'

path_unet = os.path.join(current_dir, '{}.pkl'.format(file_name_unet))  
path_aenc = os.path.join(current_dir, '{}.pkl'.format(file_name_aenc))  

with open(path_unet, 'rb') as f:
    scores_unet = pickle.load(f)

with open(path_aenc, 'rb') as f:
    scores_aenc = pickle.load(f)

snr_unet = scores_unet['snr']
pre_unet = scores_unet['precision']
rec_unet = scores_unet['recall']

snr_aenc = scores_aenc['snr']
pre_aenc = scores_aenc['precision']
rec_aenc = scores_aenc['recall']



fig, ax = plt.subplots(2, 1)
ax[0].plot(snr_unet, pre_unet, '-o', label='precision')
ax[0].plot(snr_unet, rec_unet, '-o', label='recall')
ax[0].legend()
ax[0].set_title('UNet')

ax[1].plot(snr_aenc, pre_aenc, '-o', label='precision')
ax[1].plot(snr_aenc, rec_aenc, '-o', label='recall')
ax[1].legend()
ax[1].set_title('Autoencoder')

