from PIL import Image
import glob
import os
import numpy as np

x = 184
# while x < 1392:
#     path_ini = "C:\\Users\\lucil\\OneDrive\\Bureau\\DeMarco_Lab\\data\\train" 
#     num_folder = str(x).zfill(9) 
#     path = os.path.join(path_ini, num_folder)
#     if not(path in os.listdir(path_ini)):   
#         print(path) 
#         im = Image.open(os.path.join(path, "img.png"))
#         im.save(os.path.join(path, "image.tiff"))  # or 'test.tif'
#         label = Image.open(os.path.join(path, "label.png"))
#         label.save(os.path.join(path, "label.tiff"))  # or 'test.tif'
#     x = x+1 




path_ini = "C:\\Users\\lucil\\OneDrive\\Bureau\\DeMarco_Lab\\data\\train" 
path_save = "C:\\Users\\lucil\\OneDrive\\Bureau\\DeMarco_Lab\\data\\train\\same_size" 

images_sorted = sorted(glob.glob(os.path.join(path_ini, "**\image.tif*")))
masks_sorted = sorted(glob.glob(os.path.join(path_ini, "**\label.tif*")))

for x, (im, label) in enumerate(zip(images_sorted, masks_sorted)):
    num_folder = str(x).zfill(9) 
    path = os.path.join(path_save, num_folder)  
    print(path) 
    os.mkdir(path)
    im = Image.open(im)
    im = im.resize((1024, 1536))
    #im = Image.fromarray(np.uint8(im))
    im.save(os.path.join(path, "image.tiff"))  # or 'test.tif'
    label = Image.open(label)
    label = label.resize((1024, 1536))
    #label = Image.fromarray(np.uint8(label))
    label.save(os.path.join(path, "label.tiff"))  # or 'test.tif'
