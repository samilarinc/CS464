import numpy as np
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import skimage.io
from os import listdir, system

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

folder_names = listdir("fma_small")
folder_names = folder_names[99:]
for i in folder_names:
    file_names = listdir(f"fma_small/{i}")
    system(f"mkdir mel_data/{i}")
    for j in file_names:
        y, sr = librosa.load(f"fma_small/{i}/{j}")
        imgpow = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
        imgpow = librosa.power_to_db(imgpow, ref=np.max)
        # imgpow = np.log(imgpow + 1e-9)
        np.savetxt(f"mel_data/{i}/{j[:-4]}.txt", imgpow)
        # img = scale_minmax(imgpow, 0, 255).astype(np.uint8)
        # img = np.flip(img, axis=0) 
        # img = 255-img 
        # skimage.io.imsave(f"mels/{i}/{j[:-4]}.png", img)

