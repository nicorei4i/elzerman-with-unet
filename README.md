# elzerman-with-unet

- denoising to recognize blibs in elzerman traces 
- comparison of the simple convolutional autoencoder and the unet model
- includes slurm scripts to run on high performance cluster 

- create data with 'simulate_elzerman_data.py' 
- train and test models with 'test_snr_aenc.py' or 'test_snr_unet.py'
- denoise real data with 'train_for_exdata.py' and evaluate the n_blip statistics with 'evaluate_with_exdata.py'
- measurements have to be sliced with Elzerdata.py

