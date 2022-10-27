# OpenFIBSEM
Python API for advanced FIBSEM control

## Design
- Designed to act as lego blocks, users can mix and match
- Focus on the microscopy, not learning the API
- Can script together a workflow in a few hours. 
- Can abstract common functionality, and automate manual tasks.

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
from fibsem.structures import BeamType

# connect to microscope
microscope = utils.connect_to_microscope(ip_address="10.0.0.1")

# set imaging settings
image_settings = ImageSettings(
        hfw=150e-6, 
        resolution="1536x1024",
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


## Getting Started

To get started, see the example/example.py:

(Note: You might need to edit fibsem/config/system.yaml to change the IP address of your microscope.)

This example shows you have to connect to the microscope, and take an image with both beams, and then plot.

For more detailed examples, see the Examples sections below.


## Examples

The example directory contains 

Autolamella
- Recreation of https://github.com/DeMarcoLab/autolamella, in ~150 lines of code.

Slice and View
- Recreation of a slice and view program in ~50 lines of code.

Lithography
- An example of how to perform the milling of a micro-lens in Lithium Niabate substrate using a bitmap pattern.




## Docs

Overview of the individual modules:

acquire
- helper functions for setting imaging settings, acquiring images, and post-processing.

alignment
- automated alignments of the stage and beams based fourier cross-correlation

calibration
- automated calibration routines, and microscope state management

conversions
- standard conversions between coordinate systems used in FIBSEM

milling
- helper functions for setting up, running, and finishing ion beam milling
- patterning examples

movement
- movement functionality for the stage and needle, corrected for both the view perspective and stage orientation

structures
- fibsem structures for settings, types, etc.

utils
- general microscope utilities including network connection and filesystem access 

validation
- automated validation of microscope settings (based on user configuration)

detection
- automated detection for common fibsem classes, e.g. NeedleTip, LandingPost

segmentation
- deep learning based segmentation workflow, including labelling, training and inference 

imaging
- helper functions for manipulating, analysing and masking images

ui
- user interface windows for interacting with fibsem, e.g. movement window, detection window...





enjoy