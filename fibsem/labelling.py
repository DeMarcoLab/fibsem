import glob
import napari
import numpy as np
import zarr
import tifffile as tff
import os
from PIL import Image
from tqdm import tqdm

img_dir = r""
save_dir = r""

data_path = r"C:\Users\lucil\OneDrive\Bureau\DeMarco_Lab\dm-embryo-3_20220719.104850\73d5fae8-9df6-4cc8-91e3-e2da1b94f708"
#label_path = r"C:\Users\lachl\OneDrive\Desktop\DeMarco\data\train\**\label"
save_path = r"C:\Users\lucil\OneDrive\Bureau\DeMarco_Lab\dm-embryo-3_20220719.104850\73d5fae8-9df6-4cc8-91e3-e2da1b94f708\segmentation"
zarr_path = r"C:\Users\lucil\OneDrive\Bureau\DeMarco_Lab\dm-embryo-3_20220719.104850\73d5fae8-9df6-4cc8-91e3-e2da1b94f708\segmentation\zarr"

def import_images(path: str) -> zarr.Array:
    vol = tff.imread(os.path.join(path, "*.tif*"), aszarr=True) # loading folder of .tif into zarr array)
    imgs = zarr.open(vol)
    return imgs

def label_images(save_dir: str, img_dir: str, zarr_set: zarr.Array) -> None:

    filenames = sorted(glob.glob(os.path.join(img_dir, "*.tif*")))
    # while i <= zarr_set.size:
    for img, fname in zip(zarr_set, filenames):
        print(fname)
        viewer = napari.view_image(img)
        # manually add label layer then use paint tool for segmentation
        # use different colour for different types of object. MAKE SURE TO BE CONSISTENT

        napari.run()

        # screenshot = ImageGrab.grabclipboard()
        # if screenshot is None:
        #     print("You forgot to copy image to clipboard.")

        # Saves an img with the keypoints superimposed.
        # viewer.layers.save(os.path.join(save_dir, os.path.basename(fname)))
        os.makedirs(os.path.join(save_dir, os.path.basename(fname).split(".")[0]))
        viewer.layers["img"].save(os.path.join(save_dir, os.path.basename(fname), "image"))
        viewer.layers["label"].save(os.path.join(save_dir, os.path.basename(fname), "label"))

        # i = i +1  b

def save_zarr_dataset(data_path: str, save_path: str, img_size = (1024,1536)) -> None:
    images = []
    masks = []
    sorted_img_filenames = sorted(glob.glob(data_path + ".png"))  #[-435:]
    sorted_mask_filenames = sorted(glob.glob(data_path + ".png"))  #[-435:]

    for img_fname, mask_fname in tqdm(
        list(zip(sorted_img_filenames, sorted_mask_filenames))
    ):
        image = np.asarray(Image.open(img_fname).resize(img_size))
        mask = np.asarray(Image.open(mask_fname).resize(img_size))

        images.append(image)
        masks.append(mask)

    zarr.save(os.path.join(save_path, "images.zarr"), np.array(images))
    zarr.save(os.path.join(save_path, "masks.zarr"), np.array(masks))

imgs = import_images(data_path)
label_images(save_dir, imgs)
save_zarr_dataset(save_path, save_path)
