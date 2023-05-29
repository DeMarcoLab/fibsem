# Labelling UI Guide

## Labelling UI widget

The labelling widget is built into OpenFIBSEM and provides an image processing tool to prepare labels for the purposes of training an segmentation machine learning model such as UNet

![Labelling UI](img/ml/ui_label_step.png)

## Labelling UI workflow

To begin, launch the UI and load the directories in which the images are that are to be labelled and the directory in which the labels are to be saved. Clicking the button with the three dots will open a file explorer window to select the directory. Once the paths are set, click the "Load Data" button to load the images and labels.

The number of classes refers to the unique number of objects to be segmented. In our example, we have the needle and lamella which are two classes. The number of classes can be changed at any time and the UI will update accordingly.



![Labelling UI](img/ml/select_path.png)

