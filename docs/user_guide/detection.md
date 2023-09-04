# Feature Detection and Machine Learning Tools

OpenFIBSEM hosts a variety of machine learning tools incorporated into the workflows for lamella preparation. This includes feature detection and classification through image segmentation and detection. 

Through the use of detection and classification, OpenFIBSEM can be used to automate the process of lamella preparation. This is done by detecting the features of interest and classifying them into the relevant categories. This allows for the process to make decisions on its own regarding movement and milling ultimately reducing the need for human input. 

OpenFIBSEM provides a baseline PyTorch model available on [huggingface](https://huggingface.co/patrickcleeve/openfibsem-baseline) for this purpose. This model has had  training on cryo biological samples. However, for best results, training on specific use case data is recommended. To serve this purpose, OpenFIBSEM is also constantly using the data collected to train the model to improve its accuracy. This process is further detailed below.

## Model selection

By default, if a model is not specified, the baseline model is used. This is done by loading the model from the huggingface repository. However, a model can also be specified or passed in by setting up the path of the model. 

a model can be loaded like so by using the load_model function found in fibsem/segmentation/model.py

```python

from fibsem.segmentation.model import load_model

model_path = "path/to/model" # Using model saved on disk
encoder = "resnet18" # encoder used in model, default is resnet18

model_on_disk = load_model(checkpoint = model_path, encoder = encoder)

# using model from huggingface

# if no checkpoint is passed in, the default is obtained from huggingface and saved locally
# by default, the resnet18 encoder is used

default_model = load_model()


```






## Feature Detection in Automated Lamella Preparation

Segmentation and feature detection is used in the automated lamellae preparation process to guide movement and milling. The regions of interest are segmented and detected, which are then used to control or make changes to position and milling parameters.

![detection ex](../img/user_guide/detection/detection_ex.png)

In the example shown above, the user is at the stage to mill the undercut for a lamella. The user is also supervising the workflow which ensures that the user has the final decision on the detection and feature position. 

When repositioning to mill the undercut, the system runs feature detection to identify the position of the lamella centre to ensure accurate movement of the stage. In the image, the segmentation and detection of the lamella is shown. The feature of interest being the lamella centre is set as a point in the napari viewer.

The user is then prompted to verify the detection and position of the feature. Here, if the position is invalid or incorrect, the user can simply drag the point to the correct location and press continue to proceed.

This process of user verification is enabled when the workflow is run in supervised mode on the selected stages. When the workflow is run in unsupervised mode, the model is used to control the workflow without verification from the user. This is the ideal case when the model has been trained to a point user verification is not required, further reducing human involvement in the process.


## Feedback and Model Training

When in supervised mode, the user is prompted to verify the detection and position of the feature. In the background, the supervised actions performed are used to train the model. 

I.e. when the detection is accurate and the user does not make any changes, this data is used to create training samples for the model. 

Based on whether the detection was correct or not, the images are split into a training and validation dataset. The training dataset is used to train the model and the validation dataset is used to evaluate the model.

** load model code snippet to show how hugging face model or local model is used **

When the user makes changes to the detection, this is also used to analyse and train the model. Metrics such as how far the detection was from the user corrected position is used to analyse the performance and retrain the model if and when necessary. The learning and training process is integrated within the use case of the program itself. 


As large and case base datasets are usually the bottlenecks in a machine learning workflow, the openFIBSEM program is constantly collecting training and validation data which allows for datasets to be created very easily for the purposes of training. 






 


