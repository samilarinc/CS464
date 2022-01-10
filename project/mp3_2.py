import numpy as np
import librosa
import cv2
import skimage.io

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

for i in range(10):
    y, sr = librosa.load("fma_small/000/000002.mp3", offset = 3*i, duration = 3)
    imgpow = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
    imgpow = np.log(imgpow + 1e-9)
    img = scale_minmax(imgpow, 0, 255).astype(np.uint8)
    img = np.flip(img, axis=0) 
    img = 255-img 
    skimage.io.imsave(f"deneme{i}.png", img)

y, sr = librosa.load("fma_small/000/000002.mp3")
imgpow = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
imgpow = np.log(imgpow + 1e-9)
img = scale_minmax(imgpow, 0, 255).astype(np.uint8)
img = np.flip(img, axis=0) 
img = 255-img 
skimage.io.imsave(f"deneme.png", img)
