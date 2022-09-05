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

data_path = r"C:\Users\lucil\OneDrive\Bureau\DeMarco_Lab\dm-embryo-3_20220719.104850\73d5fae8-9df6-4cc8-91e3-e2da1b94f708"
#label_path = r"C:\Users\lachl\OneDrive\Desktop\DeMarco\data\train\**\label"
save_path = r"C:\Users\lucil\OneDrive\Bureau\DeMarco_Lab\dm-embryo-3_20220719.104850\73d5fae8-9df6-4cc8-91e3-e2da1b94f708\segmentation"
zarr_path = r"C:\Users\lucil\OneDrive\Bureau\DeMarco_Lab\dm-embryo-3_20220719.104850\73d5fae8-9df6-4cc8-91e3-e2da1b94f708\segmentation\zarr"

def label_images(raw_dir: str, data_dir: str) -> None:
    vol = tff.imread(os.path.join(raw_dir, "*.tif*"), aszarr=True) # loading folder of .tif into zarr array)
    zarr_set = zarr.open(vol)

    filenames = sorted(glob.glob(os.path.join(raw_dir, "*.tif*")))

    for img, fname in zip(zarr_set, filenames):
        #Check to see if already labelled; if so, skip
        if fname.split(".")[0] in os.listdir(data_dir):
            continue

        print(fname)
        viewer = napari.view_image(img)
        # manually add label layer then use paint tool for segmentation
        # use different colour for different types of object. MAKE SURE TO BE CONSISTENT

        napari.run()

        # Saves an img with the keypoints superimposed.
        os.makedirs(os.path.join(save_dir, os.path.basename(fname).split(".")[0]))
        viewer.layers["img"].save(os.path.join(data_dir, os.path.basename(fname), "image"))
        viewer.layers["label"].save(os.path.join(data_dir, os.path.basename(fname), "label"))


def save_zarr_dataset(data_dir: str, zarr_dir: str, img_size = (1024,1536)) -> None:
    images = []
    masks = []
    sorted_img_filenames = sorted(glob.glob(os.path.join(data_dir, "image.png")))  #[-435:]
    sorted_mask_filenames = sorted(glob.glob(os.path.join(data_dir, "label.png")))  #[-435:]

    for img_fname, mask_fname in tqdm(
        list(zip(sorted_img_filenames, sorted_mask_filenames))
    ):
        image = np.asarray(Image.open(img_fname).resize(img_size))
        mask = np.asarray(Image.open(mask_fname).resize(img_size))

        images.append(image)
        masks.append(mask)

    zarr.save(os.path.join(zarr_dir, "images.zarr"), np.array(images))
    zarr.save(os.path.join(zarr_dir, "masks.zarr"), np.array(masks))

if __name__ == "__main__":

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
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
        "--zarr_dir",
        help="the directory to save the zarr dataset to",
        dest="zarr_dir",
        action="store",
        default="zarr",
    )
    parser.add_argument(
        "--img_size",
        help="resize image before saving to zarr",
        dest="img_size",
        action="store",
        type=tuple,
        default=(1024,1536)
    )

    args = parser.parse_args()
    raw_dir = args.raw_dig
    data_dir = args.data_dir
    zarr_dir = args.zarr_dir
    img_size = args.img_size

    label_images(raw_dir, data_dir)
    save_zarr_dataset(data_dir, zarr_path, img_size=img_size)

