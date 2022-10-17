from PIL import Image
import glob
import os
from tqdm import tqdm

def convert_img_size(path_ini, img_size, path_save=None, inference=False):
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

# def convert_to_folders(img_dir, label_dir, save_path, img_extension, label_extension):
#     images_sorted = sorted(glob.glob(os.path.join(img_dir, f"*.{img_extension}")))
#     masks_sorted = sorted(glob.glob(os.path.join(label_dir, f"*.{label_extension}")))

#     for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
#         num_folder = str(x).zfill(9) 
#         path = os.path.join(save_path, num_folder)  
        
#         if not os.path.exists(path):
#             os.mkdir(path)

#         im = Image.open(im)
#         im.save(os.path.join(path, "image.tiff"))  # or 'test.tif'

#         label = Image.open(label)
#         label.save(os.path.join(path, "label.tiff"))  # or 'test.tif'

import numpy as np
def convert_to_grayscale(path_ini, path_save=None, inference=False):
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

def convert_to_grayscale_inference(data_dir, save_dir=None):
    images_sorted = sorted(glob.glob(os.path.join(data_dir, "*.tif*")))

    if save_dir == None:
        save_dir = data_dir

    for x, im in enumerate(images_sorted):
        save_name = str(x).zfill(5) 
        path = save_dir
        
        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im = Image.fromarray(np.array(im)[:,:,0])
        im.save(os.path.join(path, f"{save_name}.tiff"))  # or 'test.tif'

def pad_inference(data_dir, save_dir=None):
    images_sorted = sorted(glob.glob(os.path.join(data_dir, "*.tif*")))

    if save_dir == None:
        save_dir = data_dir

    for x, im in enumerate(images_sorted):
        save_name = str(x).zfill(5) 
        path = save_dir
        
        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im = np.array(im)

        r1, r2 = round_to_32_pad(im.shape[0])
        c1,c2 = round_to_32_pad(im.shape[1])

        im = Image.fromarray(np.pad(im,pad_width=((r1,r2),(c1,c2))))

        im.save(os.path.join(path, f"{save_name}.tiff"))  # or 'test.tif'


# convert_img_size("G:\\DeMarco\\train", [1536, 1024])
# convert_to_folders('/home/rohit_k/Documents/model_training_rfp/raw', '/home/rohit_k/Documents/model_training_rfp/labels', '/home/rohit_k/Documents/model_training_rfp/train', 'tiff', 'tiff')

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

def pad_data(data_dir):
    images_sorted = sorted(glob.glob(os.path.join(data_dir, "**", "image.tif*")))
    masks_sorted = sorted(glob.glob(os.path.join(data_dir, "**", "label.tif*")))

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_folder = str(x).zfill(9) 
        path = os.path.join(data_dir, num_folder)  
        
        if not os.path.exists(path):
            os.mkdir(path)

        im = Image.open(im)
        im = np.array(im)

        r1, r2 = round_to_32_pad(im.shape[0])
        c1,c2 = round_to_32_pad(im.shape[1])

        im = Image.fromarray(np.pad(im,pad_width=((r1,r2),(c1,c2))))

        im.save(os.path.join(path, "image.tiff"))  # or 'test.tif'

        label = Image.open(label)

        r1, r2 = round_to_32_pad(label.shape[0])
        c1,c2 = round_to_32_pad(label.shape[1])
        
        label = Image.fromarray(np.pad(label,pad_width=((r1,r2),(c1,c2))))

        label = Image.fromarray(np.pad(label,pad_width=((r1,r2),(c1,c2))))

        label.save(os.path.join(path, "label.tiff"))  # or 'test.tif'

# pad_data("/home/rohit_k/Documents/model_training_rfp/train")

def convert_labels_to_index(data_dir):
    images_sorted = sorted(glob.glob(os.path.join(data_dir, "**", "image.tif*")))
    masks_sorted = sorted(glob.glob(os.path.join(data_dir, "**", "label.tif*")))

    for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
        num_folder = str(x).zfill(9) 
        path = os.path.join(data_dir, num_folder)  
        
        if not os.path.exists(path):
            os.mkdir(path)


# convert_labels_to_index("/home/rohit_k/Documents/model_training_rfp/train")
pad_inference('/home/rohit_k/Documents/RFP_Raw_data_FULL/raw_data', '/home/rohit_k/Documents/RFP_edited')
