# OpenFIBSEM

A universal API for FIBSEM Control, Development and Automation

## Overview

OpenFIBSEM is a Python package for controlling and automating FIB/SEM microscopes. It is designed to be a universal API for FIBSEM control, development and automation. OpenFIBSEM is designed to abstract away the details of the microscope and provide a simple, intuitive interface for controlling the microscope, as well as reuseable modules for common workflows and operations. OpenFIBSEM is designed to be extensible and can be easily adapted to support new microscopes.

We currently support the [TESCAN Automation SDK](https://www.tescan.com/en/products/automation-sdk/) and [ThermoFisher AutoScript](https://www.tescan.com/en/products/autoscript/). Support for other FIBSEM systems is planned.


## Install

### Install OpenFIBSEM

Clone this repository, and checkout v0.2-stable: 

```
$ git clone https://github.com/DeMarcoLab/fibsem.git
$ git checkout origin/v0.2-stable
```

Install dependencies and package
```bash
$ cd fibsem
$ conda create -n fibsem python=3.9 pip
$ conda activate fibsem
$ pip install -e .

```

For detailed instructions on installation, and installing the commercial microscope APIs, see [Installation Guide](INSTALLATION.md).

## Getting Started

To get started, see the example/example.py:

Recommended: You can start an offline demo microscope by speciying manufacturer: "Demo" in the system.yaml file (fibsem/config/system.yaml). This will start a demo microscope that you can use to test the API without connecting to a real microscope. To connect to a real microscope, set the ip_address and manufacturer of your microscope in the system.yaml or alternatively, you can pass these arguments to utils.setup_session() directly. 

This example shows you how to connect to the microscope, take an image with both beams, and then plot.

```python
from fibsem import utils, acquire
import matplotlib.pyplot as plt

def main():

    # connect to microscope
    microscope, settings = utils.setup_session(ip_address="localhost", manufacturer="Demo")

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

This example is available as a script in example/example.py.
For more detailed examples, see the Examples sections below.

## Examples

### Core Functionality

For examples of core functionality please see:

- example/example_imaging.py: image acqusition
- example/example_movement.py: stage movement
- example/example_milling.py: drawing patterns and beam milling

### Comparison to AutoScript

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
microscope, settings = utils.setup_session(ip_address="localhost", manufacturer="Thermo")

# set imaging settings
image_settings = ImageSettings(
        hfw=150e-6, 
        resolution=[1536, 1024],
        dwell_time=1e-6, 
        autocontrast=True, 
        beam_type=BeamType.ELECTRON)

# take electron image
eb_image = acquire.new_image(microscope, image_settings)

```

## Application Demonstrations

The example directory contains the following applications:

**Autolamella (autolamella.py)**

Recreation of [AutoLamella](https://github.com/DeMarcoLab/autolamella) (automated cryo-lamella preparation) in ~150 lines of code.

**Volume Microscopy (slice_and_view.py)**

Recreation of a volume microscopy workflow (slice and view) in ~50 lines of code.

**Lithography (lithography.py)**

Milling of a grayscale lithography profile using a bitmap pattern.

## Projects using OpenFIBSEM

We are currently working on a number of projects using OpenFIBSEM. If you are using OpenFIBSEM in your research, please let us know!

- [AutoLamella v2: Automated cryo-lamella preparation](www.github.com/DeMarcoLab/autolamella)
- [AutoLiftout: Automated cryo-liftout](www.github.com/DeMarcoLab/autoliftout)
- [Salami: Volume Electron Microscopy](www.github.com/DeMarcoLab/salami)
- [Vulcan: Grayscale FIB Lithography](www.github.com/DeMarcoLab/vulcan)

## Contributing

Contributions are welcome! Please open a pull request or issue.

## Docs

OpenFIBSEM is a large package with many features. For more detailed documentation, please see the [Documentation Website](https://demarcolab.github.io/fibsem/).

## Citation

```bibtex
@article{CLEEVE2023107967,
title = {OpenFIBSEM: A universal API for FIBSEM control},
journal = {Journal of Structural Biology},
volume = {215},
number = {3},
pages = {107967},
year = {2023},
issn = {1047-8477},
doi = {https://doi.org/10.1016/j.jsb.2023.107967},
url = {https://www.sciencedirect.com/science/article/pii/S1047847723000308},
author = {Patrick Cleeve and David Dierickx and Lucile Naegele and Rohit Kannachel and Lachlan Burne and Genevieve Buckley and Sergey Gorelick and James C. Whisstock and Alex {de Marco}},
keywords = {Focused Ion Beam microscopy, Automation, Python, API, Microscopy, Controller},
abstract = {This paper introduces OpenFIBSEM, a universal API to control Focused Ion Beam Scanning Electron Microscopes (FIBSEM). OpenFIBSEM aims to improve the programmability and automation of electron microscopy workflows in structural biology research. The API is designed to be cross-platform, composable, and extendable: allowing users to use any portion of OpenFIBSEM to develop or integrate with other software tools. The package provides core functionality such as imaging, movement, milling, and manipulator control, as well as system calibration, alignment, and image analysis modules. Further, a library of reusable user interface components integrated with napari is provided, ensuring easy and efficient application development. OpenFIBSEM currently supports ThermoFisher and TESCAN hardware, with support for other manufacturers planned. To demonstrate the improved automation capabilities enabled by OpenFIBSEM, several example applications that are compatible with multiple hardware manufacturers are discussed. We argue that OpenFIBSEM provides the foundation for a cross-platform operating system and development ecosystem for FIBSEM systems. The API and applications are open-source and available on GitHub (https://github.com/DeMarcoLab/fibsem).}
}
```

enjoy :)
