#%%

import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle



def plot_comparison(file_name_unet, file_name_aenc, file_name_schmitt=None, title='Performance comparison of the models', figname='comp_fig'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_unet = os.path.join(current_dir, '{}.pkl'.format(file_name_unet))  
    with open(path_unet, 'rb') as f:
        scores_unet = pickle.load(f)

    snr_unet = scores_unet['snr']
    pre_unet = scores_unet['precision']
    rec_unet = scores_unet['recall']


    path_aenc = os.path.join(current_dir, '{}.pkl'.format(file_name_aenc))  
    with open(path_aenc, 'rb') as f:
        scores_aenc = pickle.load(f)

    snr_aenc = scores_aenc['snr']
    pre_aenc = scores_aenc['precision']
    rec_aenc = scores_aenc['recall']
    
    
    if file_name_schmitt is None:
        pass
    else:
            

        path_schmitt = os.path.join(current_dir, '{}.pkl'.format(file_name_schmitt))
        with open(path_schmitt, 'rb') as f:
            scores_schmitt = pickle.load(f)
        
        snr_schmitt = scores_schmitt['snr']
        pre_schmitt = scores_schmitt['precision']
        rec_schmitt = scores_schmitt['recall']
        

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 8))

    if file_name_schmitt is None:

        ax[0].plot(snr_unet, pre_unet, '-o', label='precision')
        ax[0].plot(snr_unet, rec_unet, '-o', label='recall')
        ax[0].legend(loc='lower right')
        ax[0].set_ylabel('score')
        ax[0].set_title('UNet zoomed')
    
    else: 
        
        ax[0].plot(snr_schmitt, pre_schmitt, '-o', label='precision')
        ax[0].plot(snr_schmitt, rec_schmitt, '-o', label='recall')
        ax[1].set_ylim(0, 1.1)
        ax[0].legend(loc='lower right')
        ax[0].set_ylabel('score')
        ax[0].set_title('Schmitt Trigger')
    
    
    ax[1].plot(snr_unet, pre_unet, '-o', label='precision')
    ax[1].plot(snr_unet, rec_unet, '-o', label='recall')
    ax[1].set_ylim(0, 1.1)
    ax[1].legend(loc='lower right')
    ax[1].set_ylabel('score')
    ax[1].set_title('UNet')
    

    ax[2].plot(snr_aenc, pre_aenc, '-o', label='precision')
    ax[2].plot(snr_aenc, rec_aenc, '-o', label='recall')
    ax[2].set_ylim(0, 1.1)
    ax[2].legend(loc='lower right')
    ax[2].set_title('Autoencoder')
    ax[2].set_xlabel('snr in dB')
    ax[2].set_ylabel('score')
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(figname, format='svg', )



        

file_name_unet = 'unet_scores_4_2_3_5k'
file_name_aenc = 'aenc_scores_4_2_3_5k'
# plot_comparison(file_name_unet=file_name_unet, file_name_aenc=file_name_aenc,file_name_schmitt=None,title=r'$\Gamma_{in} = 3500$ Hz, $\Gamma_{out} = 4200$ Hz',figname='comparison_mid.svg')
plot_comparison(file_name_unet=file_name_unet, file_name_aenc=file_name_aenc,file_name_schmitt=None,title=r'$\Gamma_{in} = 3500$ Hz, $\Gamma_{out} = 4200$ Hz',figname='comparison_mid.pdf')


file_name_unet = 'unet_scores_40k'
file_name_aenc = 'aenc_scores_40k'
# plot_comparison(file_name_unet=file_name_unet, file_name_aenc=file_name_aenc,file_name_schmitt=None,title=r'$\Gamma_{in} = \Gamma_{out} = 40$ kHz',figname='comparison_high.svg')
plot_comparison(file_name_unet=file_name_unet, file_name_aenc=file_name_aenc,file_name_schmitt=None,title=r'$\Gamma_{in} = \Gamma_{out} = 40$ kHz',figname='comparison_high.pdf')

file_name_unet = 'unet_scores_400'
file_name_aenc = 'aenc_scores_400'
# plot_comparison(file_name_unet=file_name_unet, file_name_aenc=file_name_aenc,file_name_schmitt=None, title=r'$\Gamma_{in} = \Gamma_{out} = 400$ Hz',figname='comparison_low.svg')
plot_comparison(file_name_unet=file_name_unet, file_name_aenc=file_name_aenc,file_name_schmitt=None, title=r'$\Gamma_{in} = \Gamma_{out} = 400$ Hz',figname='comparison_low.pdf')


file_name_unet = 'unet_scores_real_noise_2'
file_name_aenc = 'aenc_scores_real_noise_2'
file_name_schmitt = 'schmitt_scores_real_noise_2'
# plot_comparison(file_name_unet=file_name_unet, file_name_aenc=file_name_aenc,file_name_schmitt=file_name_schmitt, title=r'$\Gamma_{in} = 3500$ Hz, $\Gamma_{out} = 4200$ Hz Real Noise',figname='comparison_mid.svg')
plot_comparison(file_name_unet=file_name_unet, file_name_aenc=file_name_aenc,file_name_schmitt=file_name_schmitt, title=r'$\Gamma_{in} = 3500$ Hz, $\Gamma_{out} = 4200$ Hz',figname='comparison_real_noise.pdf')


#%%
