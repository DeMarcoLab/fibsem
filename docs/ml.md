# Machine Learning

OpenFIBSEM contains a number of tools designed to support machine learning workflows.

## Segmentation
- segmentation model training pipeline
- used off the shelf unet models for simplicity, users can provide their own models


- train.py
- model.py
- dataset.py



## Data Labelling
- developed a user interface in napari for labelling
- compatible with fibsem images saved
- works natively with segmentation trianing pipeline

## Active Learning
- code to validate the detection in ui
- flags image for labelling
- use the labelling tools to label
- segmentation to trian
- redeploy, and repeat to improve performance
- example image