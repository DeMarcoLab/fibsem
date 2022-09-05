import glob
import napari
import numpy as np
import zarr
import tifffile as tff
import os
from PIL import Image
from tqdm import tqdm
import argparse

img_dir = r""
save_dir = r""

img_path = r"C:\Users\lachl\OneDrive\Desktop\DeMarco\data\train\**\img"
label_path = r"C:\Users\lachl\OneDrive\Desktop\DeMarco\data\train\**\label"
save_path = r"C:\Users\lachl\OneDrive\Desktop\DeMarco\fibsem segmentation"

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
        #manually add label layer then use paint tool for segmentation
        # use different colour for different types of object. MAKE SURE TO BE CONSISTENT

        napari.run()

        # screenshot = ImageGrab.grabclipboard()
        # if screenshot is None:
        #     print("You forgot to copy image to clipboard.")

        # Saves an img with the keypoints superimposed.
        # viewer.layers.save(os.path.join(save_dir, os.path.basename(fname)))
        os.makedirs(os.path.join(save_dir, os.path.basename(fname).split(".")[0]))
        viewer.layers["img"].save(os.path.join(save_dir, os.path.basename(fname), "image"))
        # i = i +1  b

def save_zarr_dataset(img_path: str, label_path: str, save_path: str, img_size = (1024,1536)) -> None:
    images = []
    masks = []
    sorted_img_filenames = sorted(glob.glob(img_path + ".png"))  #[-435:]
    sorted_mask_filenames = sorted(glob.glob(label_path + ".png"))  #[-435:]

    for img_fname, mask_fname in tqdm(
        list(zip(sorted_img_filenames, sorted_mask_filenames))
    ):
        image = np.asarray(Image.open(img_fname).resize(img_size))
        mask = np.asarray(Image.open(mask_fname).resize(img_size))

        images.append(image)
        masks.append(mask)

    zarr.save(os.path.join(save_path, "images.zarr"), np.array(images))
    zarr.save(os.path.join(save_path, "masks.zarr"), np.array(masks))

# imgs = import_images(img_dir)
# label_images(save_dir, imgs)
save_zarr_dataset(img_path, label_path, save_path)

if __name__ == "__main__":

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        help="the directory containing the raw images",
        dest="img_dir",
        action="store",
        default="images",
    )
    parser.add_argument(
        "--data_dir",
        help="the directory to save the images and labels to",
        dest="data_dir",
        action="store",
        default="data",
    )
    parser.add_argument(
        "--save_dir",
        help="the directory to save the zarr dataset to",
        dest="save_dir",
        action="store",
        default="data",
    )

    args = parser.parse_args()
    data_path = args.data
    model_checkpoint = args.checkpoint
    epochs = args.epochs
    DEBUG = args.debug
    WANDB = args.wandb