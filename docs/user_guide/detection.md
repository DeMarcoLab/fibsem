# Feature Detection and Machine Learning Tools

OpenFIBSEM hosts a variety of machine learning tools incorporated into the workflows for lamella preparation. This includes feature detection and classification through image segmentation and detection. 

Through the use of detection and classification, OpenFIBSEM can be used to automate the process of lamella preparation. This is done by detecting the features of interest and classifying them into the relevant categories. This allows for the process to make decisions on its own regarding movement and milling ultimately reducing the need for human input. 

OpenFIBSEM provides a baseline PyTorch model available on [huggingface](https://huggingface.co/patrickcleeve/openfibsem-baseline) for this purpose. This model has had baseline training on cryo biological samples. However, for best results, training on specific use case data is recommended. To serve this purpose, OpenFIBSEM is also constantly using the data collected to train the model to improve its accuracy. 

## Feature Detection in Automated Lamella Preparation

Segmentation and feature detection is used in the automated lamellae preparation process to guide movement and milling. The regions of interest are segmented and detected, which are then used to control or make changes to position and milling parameters.

![detection ex](../img/user_guide/detection/detection_ex.png)

In the example shown above, the user is at the stage to mill the undercut for a lamella. The user is also supervising the workflow which ensures that the user has the final decision on the detection and feature position. 

When repositioning to mill the undercut, the system runs feature detection to identify the position of the lamella centre to ensure accurate movement of the stage. In the image, the segmentation and detection of the lamella is shown. The feature of interest being the lamella centre is set as a point in the napari viewer.

The user is then prompted to verify the detection and position of the feature. Here, if the position is invalid or incorrect, the user can simply drag the point to the correct location and press continue to proceed.

## Feedback and Model Training

When in supervised mode, the user is prompted to verify the detection and position of the feature. In the background, the supervision actions perfomed by are used to train the model. I.e. when the detection is accurate and the user does not make any changes, this data is used to create training samples for the model. 

When the user makes changes to the detection, this is also used to analyse and train the model. Metrics such as how far the detection was from the user corrected position is used to analyse the performance and retrain the model if and when necessary. The learning and training process is ingrained within the use case of the program itself. Meaning that the model is learning and collecting data to improve itself through the use of the program itself without special input from the user.



 


