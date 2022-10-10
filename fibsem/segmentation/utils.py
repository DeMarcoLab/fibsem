from PIL import Image
import glob
import os
from tqdm import tqdm

def convert_img_size(path_ini, img_size, path_save=None):
    if path_save == None:
        path_save = path_ini

    images_sorted = sorted(glob.glob(os.path.join(path_ini, "**\image.tif*")))
    masks_sorted = sorted(glob.glob(os.path.join(path_ini, "**\label.tif*")))

    pil_size = [img_size[1], img_size[0]]

    for x, (im, label) in tqdm(enumerate(zip(images_sorted, masks_sorted))):
        num_folder = str(x).zfill(9) 
        path = os.path.join(path_save, num_folder)  

        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im = im.resize(pil_size)

        im.save(os.path.join(path, "image.tiff"))  # or 'test.tif'
        label = Image.open(label)
        label = label.resize(pil_size)

        label.save(os.path.join(path, "label.tiff"))  # or 'test.tif'

def convert_to_tiff(path_ini, img_ext, lab_ext, path_save=None):
    if path_save == None:
        path_save = path_ini

    images_sorted = sorted(glob.glob(os.path.join(path_ini, f"**\image.{img_ext}")))
    masks_sorted = sorted(glob.glob(os.path.join(path_ini, f"**\label.{lab_ext}")))

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_folder = str(x).zfill(9) 
        path = os.path.join(path_save, num_folder)  
        
        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im.save(os.path.join(path, "image.tiff"))  # or 'test.tif'

        label = Image.open(label)
        label.save(os.path.join(path, "label.tiff"))  # or 'test.tif'

def convert_to_folders(img_dir, label_dir, save_path, img_extension, label_extension):
    images_sorted = sorted(glob.glob(os.path.join(img_dir, f"*.{img_extension}")))
    masks_sorted = sorted(glob.glob(os.path.join(label_dir, f"*.{label_extension}")))

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_folder = str(x).zfill(9) 
        path = os.path.join(save_path, num_folder)  
        
        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im.save(os.path.join(path, "image.tiff"))  # or 'test.tif'

        label = Image.open(label)
        label.save(os.path.join(path, "label.tiff"))  # or 'test.tif'

# convert_img_size("G:\\DeMarco\\train", [1536, 1024])
convert_to_folders()