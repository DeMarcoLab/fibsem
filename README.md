# OpenFIBSEM
Python API for advanced FIBSEM control

## Design
- Designed to act as lego blocks, users can mix and match
- Focus on the microscopy, not learning the API
- Script together a workflow in a few hours. 
- Abstract common functionality, and automate manual tasks.

End goal is to make developing workflows for FIBSEM faster, easier, cheaper, and more accessible. 

## Example:
- Take an Electron Beam image with autocontrast...

AutoScript

```python

    from autoscript_sdb_microscope_client import SdbMicroscopeClient
    from autoscript_sdb_microscope_client.structures import GrabFrameSettings, RunAutoCbSettings

    # connect to microscope
    microscope = SdbMicroscopeClient()
    microscope.connect("10.0.0.1")

    # set the active view
    microscope.imaging.set_active_view(1)

    # set frame settings
    frame_settings = GrabFrameSettings(
        resolution="1536x1024",
        dwell_time=1e-6,
    )

    # set hfw
    microscope.beams.electron_beam.horizontal_field_width.value = 150e-6

    # autocontrast
    cb_settings = RunAutoCbSettings(
        method="MaxContrast",
        resolution="768x512",  
        number_of_frames=5,
    )
    microscope.auto_functions.run_auto_cb(cb_settings)

    # grab frame
    eb_image = microscope.imaging.grab_frame(frame_settings)

```


OpenFIBSEM

```python

from fibsem import utils, acquire
from fibsem.structures import BeamType, ImageSettings

# connect to microscope
microscope = utils.connect_to_microscope(ip_address="10.0.0.1")

# set imaging settings
image_settings = ImageSettings(
        hfw=150e-6, 
        resolution=(1536, 1024),
        dwell_time=1e-6, 
        autocontrast=True, 
        beam_type=BeamType.ELECTRON)

# take electron image
eb_image = acquire.new_image(microscope, image_settings)

```
## Install

### Install OpenFIBSEM
Clone this repository: 

```
$ git clone https://github.com/DeMarcoLab/fibsem.git
```

Install dependencies and package
```bash
$ cd fibsem
$ conda env create -f environment.yml
$ conda activate fibsem
$ pip install -e .

```

### Install AutoScript
You will also need to install AutoScript 4.6+. 

Please see the [Installation Guide](INSTALLATION.md) for detailed instructions.

Copy AutoScript /into home/user/miniconda3/envs/fibsem/lib/python3.9/site-packages/

### Install TESCAN Automation SDK

Ideally, please install and set up the conda environment first before proceeding to install this SDK

Run the Tescan-Automation-SDK-Installer-3.x.x.exe file

When asked for the python interpretor, select the existing conda environment for FIBSEM, if this python interpretor is not available, see detailed installation guide for a work around

See [Installation Guide](INSTALLATION.md) for full details

## Getting Started

To get started, see the example/example.py:

(Note: You might need to edit fibsem/config/system.yaml to change the IP address of your microscope.)

This example shows you how to connect to the microscope, take an image with both beams, and then plot.

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

For more detailed examples, see the Examples sections below.


## Examples

The example directory contains 

Autolamella
- Recreation of https://github.com/DeMarcoLab/autolamella, in ~150 lines of code.

Slice and View
- Recreation of a slice and view program in ~50 lines of code.

Lithography
- Milling of a lithography profile using a bitmap pattern.


## Docs

TODO: finish once site is up


## Citation

TODO:


enjoy :) 