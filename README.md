# fibsem
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


OpenFIB

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

# take reference image with both beams..
eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
```

More things going on under the hood: 
- Additional options in ImageSettings to auto-gamma correct, and save the image.
- Will try to prevent user errors, e.g. Won't crash when you try to take an image with larger hfw than maximum, will just clip to the maximum hfw.

## Three primary use cases

### Customising workflows
As openFIB gives complete control over how workflows run, you can customise, iterate and experiment with different strategies. It becomes simple to script workflows that would be tedious or error prone to do manually.

E.g. Slice and View: denoising data collection strategy
Take 5 fast dwell time images, then a long dwell time image...

``` python

def custom_data_collection(microscope: SdbMicroscopeClient, image_settings: ImageSettings):

    image_settings.save = True
    image_settings.dwell_time = 0.1e-9
    label = "short_dwell_time"

    for i in range(5):
        image_settings.label = f"{label}_{i}"
        acquire.new_image(microscope, image_settings)

    image_settings.dwell_time = 4e-6
    label = "long_dwell_time"

    acquire.new_image(microscope, image_settings)

```

### Automating existing workflows
E.g. Autolamella, Lens Milling

As the modules of openfib act like lego blocks, you can quickly re-write existing workflows to use the components and automated functionality. You can also mix and match with existing code and libraries easily.

See example/autolamella.py
See example/napari_visualisation.py

### Creating new workflows and tools
E.g. AutoLiftout, automated tools: auto-discharge, auto-needle-calibration

As more functionality is developed, openfib enables you to make higher level and more automated workflows. 


``` python

def auto_discharge_beam(microscope: SdbMicroscopeClient, image_settings: ImageSettings, n_iterations: int = 10):

    # take sequence of images quickly to discharge the sample
    resolution = image_settings.resolution
    dwell_time = image_settings.dwell_time
    autocontrast = image_settings.autocontrast
    beam_type = image_settings.beam_type
    
    image_settings.beam_type = BeamType.ELECTRON
    image_settings.resolution = "768x512"
    image_settings.dwell_time = 200e-9
    image_settings.autocontrast = False
        
    for i in range(n_iterations):
        acquire.new_image(microscope, image_settings)

    # autocontrast
    acquire.autocontrast(microscope, BeamType.ELECTRON)

    # restore settings, take image
    image_settings.resolution = resolution
    image_settings.dwell_time = dwell_time
    image_settings.autocontrast = autocontrast
    acquire.new_image(microscope, image_settings)
    
    image_settings.beam_type = beam_type

```



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


## Examples


Autolamella
- Recreation of https://github.com/DeMarcoLab/autolamella, in ~150 lines of code.

Slice and View
- Recreation of a slice and view program in ~50 lines of code.

Lens Milling (TODO)
- An example of how to perform the milling of a micro-lens in Lithium Niabate substrate using a bitmap pattern.

Napari Visualisation (TODO)
- An example of connecting the microscope directly into Napari to visualise images.