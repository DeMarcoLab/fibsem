# FIBSEM Segmentation

This section of the repository contains all of the code needed to label a segmentation dataset, create and train an automated segmentation model, as well as utilise said model for inference purposes.

## Getting started
1. All of the required installation steps should have already been completed in the base README.

2. (Optional) Download the sample dataset [Google Drive](Add link here)

## Running the segmentation code
The code relating to the creation and training of a segmentation model expects the dataset images and labels to be in a TIFF File format. If you are utilising the labelling code within this repository to label your dataset, this is automatically done for you. If you are using a pre-existing dataset that is not in TIFF File format, there is a helper function in dataset.py that can be used to convert images of any extension to TIFF.

Labelling is performed by labelling.py, training and validation is performed by train.py, and inference is performed by inference.py. All of these files expect a yaml config file that is used to specify the directories and parameters to be used.

### Config.yml
```
labelling:
  raw_dir: directory containing the raw data to be labelled
  data_dir: directory to save the labelled data to
train:
  data_dir: directory containing the labelled dataset
  save_dir: directory to save the model weights to
  wandb: logs information and plots to wandb if true
  checkpoint: path to model if you would like to resume training
  encoder: specify model architecture. List of available encoders in readme.
  epochs: number of epochs to train for
  cuda: enable/disable CUDA training
  batch_size: number of batches per epoch
  num_classes: number of classes in segmentation labels. Includes background as class 0.
  optimizer: adam or SGD, not case sensitive
  learning_rate: learning rate used during training
```
TODO: Add inference to config.yml

### Labelling
The code for labelling the dataset can be found in labelling.py. It expects the directory of unlabelled images and the directory to save the labelled images to be found in the config.yml file. 

To run this file from the command line:
1. cd into the segmentation directory

2. 
```
$ python labelling.py --config config.yml
```

Once labelling.py is running and the unlabelled images have been imported, a napari viewer will open with the image already loaded. To create a segmentation label:
1. Create a new 'Labels' layer
2. Select the paint brush icon, and begin painting the objects of interest. To save time it is recommended to paint the outline of each object, and then use the fill tool. 
3. If you have multiple classes, do not create a new Labels layer for each class, simply use a different index for each class in the Labels layer. NOTE: MAKE SURE YOU ARE CONSISTENT WITH THE INDEX FOR EACH CLASS.
4. When you have finished labelling an image, exit the viewer to save the image and the segmentation label to the destination you specified in the config. A new viewer will then pop up with the next image.
5. When you would like to take a break from labelling the dataset, close the napari viewer without creating a Labels layer, this will stop the script. NOTE: If you have an image that does not have any objects of interest in it, create an empty Labels layer before closing the viewer to save the image, otherwise the script will just close.
6. Once you stop the script either intentionally or accidentally, the script will begin where you left off.

### Training and Validation
The code for training and validation can be found in train.py. It expects the labelled dataset directory as well as the directory to save your trained model in the config.yml file. The rest of the options have defaults that can be changed as needed.

To run this file from the command line:
1. cd into the segmentation directory

2. 
```
$ python train.py --config config.yml
```

### Inference

