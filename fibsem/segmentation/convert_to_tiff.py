from PIL import Image
import glob
import os


x = 1099
while x < 1392:
    path_ini = "C:\\Users\\lucil\\OneDrive\\Bureau\\DeMarco_Lab\\data\\train" 
    num_folder = str(x).zfill(9) 
    path = os.path.join(path_ini, num_folder)
    if not(path in os.listdir(path_ini)):   
        print(path) 
        im = Image.open(os.path.join(path, "img.png"))
        im.save(os.path.join(path, "image.tiff"))  # or 'test.tif'
        label = Image.open(os.path.join(path, "label.png"))
        label.save(os.path.join(path, "label.tiff"))  # or 'test.tif'
    x = x+1 
