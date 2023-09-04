# Feature Detection and Machine Learning Tools

OpenFIBSEM hosts a variety of machine learning tools incorporated into the workflows for lamella preparation. This includes feature detection and classification through image segmentation and detection. 

Through the use of detection and classification, OpenFIBSEM can be used to automate the process of lamella preparation. This is done by detecting the features of interest and classifying them into the relevant categories. This allows for the process to make decisions on its own regarding movement and milling ultimately reducing the need for human input. 

OpenFIBSEM provides a baseline PyTorch model available on [huggingface](https://huggingface.co/patrickcleeve/openfibsem-baseline) for this purpose. This model has had baseline training on cryo biological samples. However, for best results, training on specific use case data is recommended. To serve this purpose, OpenFIBSEM is also constantly using the data collected to train the model to improve its accuracy. 




