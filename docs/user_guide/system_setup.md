# System Configuration and Setup Guide

This guide outlines the steps to configure and setup the system parameters. Each subsystem will be explained. This process does not need to be done everytime the system is started. A default is provided. If changes are made, this can be saved to a file and loaded on startup.

On start up of the application the configuration tab will be loaded. Here you can make changes as necessary. 

![Configuration Tab](../img/user_guide/setup/config_tab.png)

To load the default configuration and connect to the microscope, simply click the "Connect to Microscope" button in the connect tab. This will load the default configuration and connect to the microscope.

The configuration can be saved and loaded from a file. This allows for the user to save a configuration and load it on start up. To save, click "Save configuration to file" and select a location to save the file. To load, click "Import Configuration" and select the file to load. The file to load must be a configuration type yaml file that has been previously saved.

The configuration parameters can be altered before connection to the microscope. To make changes to the microscope configuration while it is connected, click "Apply All Settings" to apply the changes. This will update the parameters in the program.

## Parameters

### Stage Settings

The stage parameters can be defined in this section. This section outlines the basic stage and beam column angles which allows for correct calculations of the stage position and subsequently, the movement of the stage.

![Stage Parameters](../img/user_guide/setup/stage_params.png)

### Microscope Model Settings

The microscope parameters can be defined in this section. This section specifies the main hardware inclusions available on the microscope. This lets the user enable or disable functionality depending on availablity. For example, if a system does not have a manipulator, this can be disabled here and other functionality can be used without errors

![Microscope Parameters](../img/user_guide/setup/microscope_params.png)

### System Settings

The system parameters related to the ion and electron beams can be defined and set up in this section. In this section, you can also specify if the system is to be connected directly on start up or manually connect to the microscope each time.

![System Parameters](../img/user_guide/setup/system_params.png)

### User Settings

The user parameters related to milling and imaging  can be specified in this section. 

![User Parameters](../img/user_guide/setup/user_params.png)



 

