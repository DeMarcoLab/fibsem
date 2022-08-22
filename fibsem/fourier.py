# TODO
import numpy as np
import logging
from scipy import fftpack

from fibsem.imaging import masks


def crosscorrelation(img1: np.ndarray, img2: np.ndarray,  
    lp: int = 128, hp: int = 6, sigma: int = 6, bp: bool = False) -> np.ndarray:
    """Cross-correlate images (fourier convolution matching)

    Args:
        img1 (np.ndarray): reference_image
        img2 (np.ndarray): new image
        lp (int, optional): lowpass. Defaults to 128.
        hp (int, optional): highpass . Defaults to 6.
        sigma (int, optional): sigma (gaussian blur). Defaults to 6.
        bp (bool, optional): use a bandpass. Defaults to False.

    Returns:
        np.ndarray: crosscorrelation map
    """
    if img1.shape != img2.shape:
        err = f"Image 1 {img1.shape} and Image 2 {img2.shape} need to have the same shape"
        logging.error(err)
        raise ValueError(err)

    if bp: 
        bandpass = masks.bandpass_mask(
            size=(img1.shape[1], img1.shape[0]), 
            lp=lp, hp=hp, sigma=sigma
        )
        n_pixels = img1.shape[0] * img1.shape[1]
        
        img1ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img1)))
        tmp = img1ft * np.conj(img1ft)
        img1ft = n_pixels * img1ft / np.sqrt(tmp.sum())
        
        img2ft = fftpack.ifftshift(bandpass * fftpack.fftshift(fftpack.fft2(img2)))
        img2ft[0, 0] = 0
        tmp = img2ft * np.conj(img2ft)
        
        img2ft = n_pixels * img2ft / np.sqrt(tmp.sum())

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2, figsize=(15, 15))
        # ax[0].imshow(fftpack.ifft2(img1ft).real)
        # ax[1].imshow(fftpack.ifft2(img2ft).real)
        # plt.show()

        xcorr = np.real(fftpack.fftshift(fftpack.ifft2(img1ft * np.conj(img2ft))))
    else: # TODO: why are these different...
        img1ft = fftpack.fft2(img1)
        img2ft = np.conj(fftpack.fft2(img2))
        img1ft[0, 0] = 0
        xcorr = np.abs(fftpack.fftshift(fftpack.ifft2(img1ft * img2ft)))
    
    return xcorr

# numpy version
def crosscorrelation_v2_np(img1: np.ndarray, img2: np.ndarray,  
    lp: int = 128, hp: int = 6, sigma: int = 6, bp: bool = False) -> np.ndarray:
    """Cross-correlate images (fourier convolution matching)

    Args:
        img1 (np.ndarray): reference_image
        img2 (np.ndarray): new image
        lp (int, optional): lowpass. Defaults to 128.
        hp (int, optional): highpass . Defaults to 6.
        sigma (int, optional): sigma (gaussian blur). Defaults to 6.
        bp (bool, optional): use a bandpass. Defaults to False.

    Returns:
        np.ndarray: crosscorrelation map
    """
    if img1.shape != img2.shape:
        err = f"Image 1 {img1.shape} and Image 2 {img2.shape} need to have the same shape"
        logging.error(err)
        raise ValueError(err)

    if bp: 
        bandpass = masks.bandpass_mask(
            size=(img1.shape[1], img1.shape[0]), 
            lp=lp, hp=hp, sigma=sigma
        )
        n_pixels = img1.shape[0] * img1.shape[1]
        
        img1ft = np.fft.ifftshift(bandpass * np.fft.fftshift(np.fft.fft2(img1)))
        tmp = img1ft * np.conj(img1ft)
        img1ft = n_pixels * img1ft / np.sqrt(tmp.sum())
        
        img2ft = np.fft.ifftshift(bandpass * np.fft.fftshift(np.fft.fft2(img2)))
        img2ft[0, 0] = 0
        tmp = img2ft * np.conj(img2ft)
        
        img2ft = n_pixels * img2ft / np.sqrt(tmp.sum())

        xcorr = np.real(np.fft.fftshift(np.fft.ifft2(img1ft * np.conj(img2ft))))
    else: # TODO: why are these different...
        img1ft = np.fft.fft2(img1)
        img2ft = np.conj(np.fft.fft2(img2))
        img1ft[0, 0] = 0
        xcorr = np.abs(np.fft.fftshift(np.fft.ifft2(img1ft * img2ft)))
    
    return xcorr