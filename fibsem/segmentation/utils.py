from PIL import Image
import glob
import os
from tqdm import tqdm
import numpy as np

def convert_img_size(path_ini, img_size, path_save=None, inference=False):
    """converts image to defined image size"""
    if path_save == None:
        path_save = path_ini

    if not inference:
        masks_sorted = sorted(glob.glob(os.path.join(path_ini, "labels", "*.tif*")))

    images_sorted = sorted(glob.glob(os.path.join(path_ini, "images", "*.tif*")))
    

    pil_size = [img_size[1], img_size[0]]

    for x, (im, label) in tqdm(enumerate(zip(images_sorted, masks_sorted))):
        num_file = str(x).zfill(9) 
        path = path_save

        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im = im.resize(pil_size)
        im.save(os.path.join(path, "images", f"{num_file}.tiff"))  # or 'test.tif'

        if not inference:
            label = Image.open(label)
            label = label.resize(pil_size)
            label.save(os.path.join(path, "labels", f"{num_file}.tiff"))  # or 'test.tif'

def convert_to_tiff(path_ini, img_ext, lab_ext, path_save=None, inference=False):
    """converts images to tiff"""
    if path_save == None:
        path_save = path_ini

    images_sorted = sorted(glob.glob(os.path.join(path_ini, "images", f"*.{img_ext}")))
    if not inference:
        masks_sorted = sorted(glob.glob(os.path.join(path_ini, "labels", f"*.{lab_ext}")))

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_file = str(x).zfill(9) 
        path = path_save
        
        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im.save(os.path.join(path, "images", f"{num_file}.tiff"))  # or 'test.tif'

        if not inference:
            label = Image.open(label)
            label.save(os.path.join(path, "labels", f"{num_file}.tiff"))  # or 'test.tif'


def convert_to_grayscale(path_ini, path_save=None, inference=False):
    """converts images to grayscale"""
    images_sorted = sorted(glob.glob(os.path.join(path_ini, "images", "*.tif*")))

    if not inference:
        masks_sorted = sorted(glob.glob(os.path.join(path_ini, "labels", "*.tif*")))

    if path_save == None:
        path_save = path_ini

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_file = str(x).zfill(9) 
        path = path_save
        
        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im = Image.fromarray(np.array(im)[:,:,0])
        im.save(os.path.join(path, "images", f"{num_file}.tiff"))  # or 'test.tif'

        if not inference:
            label = Image.open(label)
            label.save(os.path.join(path, "labels", f"{num_file}.tiff"))  # or 'test.tif'


def round_to_32_pad(num:int)->tuple[int,int]:
    """Rounds up an integer to the nearest multiple of 32. The difference
    between the number and its nearest multiple of 32 is then split in half
    and provided. This function is used to easily calculate padding values 
    when padding images for suitability with PyTorch NN

    e.g. if num == 60
    then closest is 64 and two values returned will be 2 and 2
    if num == 61, then values returned will be 1 and 2

    Args:
        num (int): value of dimension

    Returns:
        tuple[int,int]:  
    """

    m1 = -(-num//32)
    m2 = 32*m1

    val = m2 - num
    if val % 2 == 0:
        x1,x2 = val/2,val/2
    else:
        x1 = round(val/2)
        x2 = val - x1

    return int(x1),int(x2)

def pad_data(path_ini, path_save=None, inference=False):
    """converts image size to multiple of 32"""
    images_sorted = sorted(glob.glob(os.path.join(path_ini, "images", "*.tif*")))

    if not inference:
        masks_sorted = sorted(glob.glob(os.path.join(path_ini, "labels", "*.tif*")))

    if path_save == None:
        path_save = path_ini

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_file = str(x).zfill(9) 
        path = path_save
        
        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im = np.array(im)

        r1, r2 = round_to_32_pad(im.shape[0])
        c1,c2 = round_to_32_pad(im.shape[1])

        im = Image.fromarray(np.pad(im,pad_width=((r1,r2),(c1,c2))))
        im.save(os.path.join(path, "images", f"{num_file}.tiff"))  # or 'test.tif'

        if not inference:
            label = Image.open(label)
            label = np.array(label)

            r1, r2 = round_to_32_pad(label.shape[0])
            c1,c2 = round_to_32_pad(label.shape[1])
            
            label = Image.fromarray(np.pad(label,pad_width=((r1,r2),(c1,c2))))
            label.save(os.path.join(path, "labels", f"{num_file}.tiff"))  # or 'test.tif'


def convert_folder_format(directory, path_save):
    """converts folder format to correct format"""
    images_sorted = sorted(glob.glob(os.path.join(directory, "**", "image.tif*")))
    masks_sorted = sorted(glob.glob(os.path.join(directory, "**", "label.tif*")))

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_file = str(x).zfill(9) 
        path = path_save
        
        os.mkdir(os.path.join(path, "images"), exist_ok=True)
        os.mkdir(os.path.join(path, "labels"), exist_ok=True)

        im = Image.open(im)
        im.save(os.path.join(path, "images", f"{num_file}.tiff"))  # or 'test.tif'

        label = Image.open(label)
        label.save(os.path.join(path, "labels", f"{num_file}.tiff"))  # or 'test.tif'