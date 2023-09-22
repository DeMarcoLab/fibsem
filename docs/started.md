# Getting Started


### Installation

Please see README.md

### Desktop Shortcut

To create a desktop shortcut for FibsemUI, simply run the shortcut.py file.

### System Configuration

The system configuration is defined by a system.yaml file. This defines various system settings, including connection details, beam, imaging, and stage settings. 

You will need to change the ip_address to connect to your microscope. 

The default file can be found in fibsem/config/system.yaml. When you call utils.setup_session() with no arguments, the default file is used. You can either edit the default file, or provide the config_path to your own system.yaml file in setup_session.

WIP: update for v2

```yaml
# system
system:
  ip_address: 10.0.0.1
  application_file: autolamella
  manufacturer: Thermo # Thermo or Tescan
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
    eucentric_height: 4.0e-3
    detector_type: ETD
    detector_mode: SecondaryElectrons
  # stage
  stage:
    rotation_flat_to_electron: 50 # degrees
    rotation_flat_to_ion: 230 # degrees
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

Note: setup_session will not automatically switch to these settings. To do so, you need to call validation.validate_initial_microscope_state.


### Example
Once you have changed your system.yaml file, you should be able to run example/example.py to take images with both beams, and plot.

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

## The Basics

### Microscope Connection

The microscope is a client connection to the Microscope Server. At the moment, only ThermoFisher AutoScript Client is supported. 

### MicroscopeSettings

MicroscopeSettings is a large structure containing the settings for the different microscope systems.

::: fibsem.structures.MicroscopeSettings

It is populated from your configuration in system.yaml. Most functions take a combination of the microscope client, and settings as arguments. 

### Taking an Image

The most basic use case for the package is taking an image. Imaging functions are contained in fibsem.acquire, and imaging conditions are controlled by modifying the ImageSettings struct.

::: fibsem.acquire.new_image

:::fibsem.structures.ImageSettings

ImageSettings is part of the MicroscopeSettings, and can be accessed by settings.image. For example, to change the hfw of the image and take an image (snippet):

```python

# change hfw to 400e-6
settings.image.hfw = 400e-6

# take image
image = acquire.new_image(microscope, settings.image)


```

### Movement
...

### Milling
...