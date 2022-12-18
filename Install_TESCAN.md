# TESCAN Automation SDK Installation

## Overview

This guide is to provide step by step instructions on how to install the Tescan Automation SDK to work with the FIBSEM module and TESCAN microscopes.

## Prerequisites

Before beginning this install, please ensure the following

- FIBSEM conda environment is installed and setup
- tescan-automation-sdk-install exe file is ready to go

## Installing the SDK

Run the installer exe file. When it asks for the python interpreter, select the one that is on the conda environment and proceed with the install

### Common Issue with Python Interpreter

If the conda python interpreter cannot be selected from the drop down options, proceed with the install and take note of the path of installed python interpreter. 

(If no python interpreter can be found in the drop down, install python 3.9+ seperately and run the installation exe again)

Once the installation has been completed, navigate to where python is installed on which the SDK has been installed.

In there navigate to


`.../python/lib/site-packages`

from this folder, find and copy the following folders:

- All folders beginning with `PySide6`
- All folders beginning with `shiboken`
- All folders beginning with `tescan`

Copy these into the python folder that is set up in the conda environment

`.../Anaconda3/envs/fibsem/lib/site-packages`

The package should now be installed successfully

## Checking Install

To check if the module has been installed properly and can be imported run the following python code in FIBSEM:

```python
import sys
from tescanautomation import Automation

print("Tescan Imported Successfully") if "tescanautomation" in sys.modules else print("Tescan Import was unsuccessful")

```
