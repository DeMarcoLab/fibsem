# OpenFIBSEM Documentation

## Overview
OpenFIBSEM is a platform built with python to allow high level FIB-SEM microscope control using a single package across multiple supported microscopes. Currently, support for microscopes of Thermo Fisher Scientific and TESCAN are actively being developed. 

The OpenFIBSEM package allows control of the microscope in the same syntactic manner regardless of the brand of microscope in use as the package is built upon the existing SDKs for both the different microscopes. 

The microscope client is set up using an internet connection. Hence the microscope control PC I.P. address needs to be set to connect. (“localhost” if on a PC connected directly to a microscope).

The microscope organisation is abstracted away in the FibsemMicroscope class. All hardware implementations follow the same abstract organisation. This allows for a black box approach in the external modules, where all calls to the microscope are independent of the specific hardware, and all direct calls to the APIs are limited to the class methods. 

  
![Figure 1: Implemented methods in microscope class](/docs/openFibsemv2.png)

As an example, to take a set of reference images from the microscope using an electron beam and an ion beam, the syntax is as follows:

    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)

In this case, the type of microscope is not relevant for the user and this method will work. However, the type of microscope must be initialised during setup. If the microscope in use is a supported Thermo Fisher or TESCAN microscope, it must be set up as so in the system.yaml file.

Along with taking images, specific imaging parameters can also be specified. This can be done by creating and initialising an image settings object.

## Setting up Microscope

To set up a session with a microscope, relevant microscope type and connection parameters need to be set up in the system.yaml config file that is located in the following directory in the package:

    …/FIBSEM/fibsem/config/system.yaml



In the yaml file, the connection set up parameters required are
- I.P. Address
   - Address of the microscope to be used. The text “localhost” is valid if the microscope to connect to is on the local pc
- Manufacturer
    - Microscope manufacturer, If the microscope brand is supported by its relevant SDK packages.

```yaml 
# system
system:
  ip_address: "localhost" 
  application_file: autolamella
  manufacturer: "Thermo"
  # beams
  ion:
    voltage: 30000
    current: 20.e-12
    plasma_gas: "Argon" # proper case, e.g. Argon, Oxygen
    eucentric_height: 16.5e-3
    detector_type: ETD
    detector_mode: SecondaryElectrons
  electron:
    voltage: 2000
    current: 1.0e-12
    eucentric_height: 3.91e-3
    detector_type: ETD
    detector_mode: SecondaryElectrons
  # stage
  stage:
    rotation_flat_to_electron: 49 # degrees
    rotation_flat_to_ion: 229 # degrees
    tilt_flat_to_electron: 35 # degrees (pre-tilt)
    tilt_flat_to_ion: 52 # degrees
    pre_tilt: 35
    needle_stage_height_limit: 3.7e-3
# user config settings
user:
  imaging_current: 20.e-12
  milling_current: 2.e-9
  resolution: "1536x1024"
  hfw: 150.e-6  
  pixel_size: 
    x: 0.0
    y: 0.0
  beam_type: "Electron"
  autocontrast: True
  dwell_time: 1.e-6
  save: False
  gamma: # gamma correction parameters
    enabled: True
    min_gamma: 0.15
    max_gamma: 1.8
    scale_factor: 0.01
    threshold: 46 # px
```


User parameters can also be preset using the system.yaml file. All numerical values are in SI units: 
- Current: Amps (A)
- Time: Seconds (s)
- Distance: Metres (m)

# Setting Hardware Parameters

The hardware specifications need to be provided in the model.yaml file, found at:

      …/FIBSEM/fibsem/config/model.yaml

```yaml
# this file is used to define the microscope model, and enabled components
system:
  name: "FIBSEM"
  manufacturer: "Demo"
  description: "FIBSEM"
  version: "0.1"
  id: 00000
# define the microscope model
electron:
  enabled: True
ion:
  enabled: True
stage:
  enabled: True
  rotation: True
  tilt: True
manipulator:
  enabled: True
  rotation: True
  tilt: True
gis:
  enabled: True
  multichem: True
```
Simply set the parameter to True if the hardware is present in the microscope. 

## Example code
An example function for taking images with both beams can be found in the example_OpenFIBSEM file. 

```python
from fibsem import utils, acquire
import matplotlib.pyplot as plt


def main():
    # connect to microscope
    microscope, settings = utils.setup_session()

    # take image with both beams
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)

    # show images
    fig, ax = plt.subplots(1, 2, figsize=(7, 5))
    ax[0].imshow(eb_image.data, cmap="gray")
    ax[1].imshow(ib_image.data, cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()
```

![Figure 2: Example output ](/docs/Figure_1.png)



