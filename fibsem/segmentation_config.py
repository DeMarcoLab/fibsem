import json

# Add you directories here
RAW_DIR = r"C:\Users\lachl\OneDrive\Desktop\DeMarco\raw_img"
DATA_DIR = r"C:\Users\lachl\OneDrive\Desktop\DeMarco\data_img"
ZARR_DIR = r"C:\Users\lachl\OneDrive\Desktop\DeMarco\zarr_img"

segmentation_config = {
    "raw_dir": RAW_DIR,
    "data_dir": DATA_DIR,
    "zarr_dir": ZARR_DIR
}

# Saves a JSON file that git will ignore to allow each user to use local directories
with open("segmentation_config.json", 'w') as f:
    json.dump(segmentation_config, f)

