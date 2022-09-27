from PIL import Image
import glob
import os

def convert_img_size(path_ini, img_size, path_save=None):
    if path_save == None:
        path_save = path_ini

    images_sorted = sorted(glob.glob(os.path.join(path_ini, "**\image.tif*")))
    masks_sorted = sorted(glob.glob(os.path.join(path_ini, "**\label.tif*")))

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_folder = str(x).zfill(9) 
        path = os.path.join(path_save, num_folder)  

        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im = im.resize(img_size)

        im.save(os.path.join(path, "image.tiff"))  # or 'test.tif'
        label = Image.open(label)
        label = label.resize(img_size)

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
