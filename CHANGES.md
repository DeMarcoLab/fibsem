# Changes

## v0.4.0 (10/02/2025)

Current Status: Pre-Release

## Installation
- The minimum required python version is now 3.8 (down from 3.9). This should enable installing fibsem on older systems that cannot be updated from Windows 7. 
- Both fibsem and autolamella can now be run in 'headless' mode without the UI (or requiring it's dependencies). This is used for embedding openfibsem into other standalone applications. 
 - The machine learning dependencies (used for more advanced methods) are now optional.
- Instaling packages is now slightly different to reflect these optional dependencies:

```
pip install fibsem          # only install fibsem headless mode
pip install fibsem[ui]      # install fibsem + ui dependencies
pip install fibsem[ui,ml]   # install fibsem + ui and ml dependencies
```

## Milling
- The milling code has been consolidated into the fibsem.milling module.

```

fibsem
    milling
        base.py             # base milling structures 
        core.py             # core milling workflow
        patterning/
            patterns.py     # pattern definitions
            plotting.py     # plotting utilities
        strategy/
            standard.py    # standard (default) milling strategy
            ...            # additional strategy files
```

### Patterns
- Milling patterns directly store parameters, instead of reading in a protocol dictionary.
- Some patterns have had their parameter names adjusted for clarity and generality:


#### Trench, HorseshoePattern, HorseshoeVertical
- Trench based parameters have been adjusted:
- lamella_width -> width
- lamella_height -> height
- size_ratio -> split into upper_trench_height, lower_trench_height
- Loading an older protocol in autolamella should automatically convert to the new format. If older protocols are not read correctly, consider it a bug and please get in contact.

#### RectangleOffset
- RectangleOffset patterns have been removed, as their purpose was to position rectangle patterns via the protocol. The position of patterns can be directly specified in prootocol, using point: {x, y}

### Milling Stage

- Estimated milling time can now be calculated independently from the microscope. This will be less accurate than the real duration calculated during milling.
- Milling stages now have additional configuration options:

#### Milling Strategy
-  Previously, openfibsem only supported a basic milling process; Set milling settings, draw patterns, mill patterns, restore imaging settings. Milling strategies enable customising the milling process. 
- Currently only Standard and Overtilt milling are implemented, but more will be added in the future.
- Developers can add additional strategies by implementing the spec in milling.base. Additional strategies can be registered by:
  - Registering them using a plugin-style registry
    ```
    from fibsem.milling.strategy import register_strategy
    from custom_strategy import CustomMillingStrategy

    register_strategy(CustomMillingStrategy)

    ```
  - Creating a `fibsem.strategies` entrypoint that points to the strategy class and installing the package in the same environment as fibsem, e.g. in pyproject.toml:
    ```
    [project.entry-points.'fibsem.strategies']

    # Make the strategy discoverable by fibsem
    # The class CustomMillingStrategy is in my_pkg/strategy.py

    strategy = "my_pkg.strategy:CustomMillingStrategy"

    ```

#### Milling Alignment
- Previously, aligning milling currents was only available via the autolamella option (align_at_milling_current). This was not straightforward to use or easily discoverable. 
- Initial milling alignment is now available for each stage. This will acquire an image after changing to the milling current and re-align to the imaging current. By default it will use the alignment area (fiducial area) defined in autolamella.
- Interval based drift correction will be enabled in the next version (v0.4.1)

#### Milling Acquisition
- You can now specify to acquire an image at the end of each milling stage. The acquisition settings can be adjusted per stage.

### User Interface:
- Parameters now display units directly on the control (rather than the label)
- Tooltips are being added to UI elements to help explain different parameters and options. 
- Acquire Image has been split into individual channels (Acquire SEM/ Acquire FIB)
- You can now show/hide milling patterns in the UI.
- You can now pause and resume milling from the UI.
- You can now select individual stages to mill, rather than having to mill all at once.
- Advanced options have been added to the imaging UI (e.g. line integration)


### Developer Notes:
- New tools are available for debugging milling patterns:

```
import matplotlib.pyplot as plt

from fibsem import utils
from fibsem.milling import get_milling_stages
from fibsem.milling.patterning.plotting import draw_milling_patterns
from fibsem.structures import FibsemImage

# load protocol
PROTOCOL_PATH = "/path/to/protocol/protocol-on-grid.yaml"
protocol =  utils.load_protocol(PROTOCOL_PATH)

# get the milling stages
stages = get_milling_stages("mill_rough", settings.protocol["milling"]) 
stages.extend(get_milling_stages("mill_polishing", settings.protocol["milling"]))
stages.extend(get_milling_stages("microexpansion", settings.protocol["milling"]))

# create a blank image
image = FibsemImage.generate_blank_image(hfw=stages[0].milling.hfw)

# plot the milling stages
fig = draw_milling_patterns(image, stages)
plt.show()
```

- More milling data is now logged at each stage, and can be exported to run analysis. The milling related data is exported in milling.csv (see AutoLamella v0.4.0)


### Experimental Features
- There is now an experimental writer for exporting openfibsem images in OME-TIFF format. This will be enabled as default in the next version (v0.4.1). This should enable other applications  (e.g. ImageJ) to open the images and read the metadata correctly.


## v0.2.2 - 31/07/2023

### Highlights

- OpenFIBSEM is now available on PyPI. Use pip to install: `pip install fibsem`. On ThermoFisher systems, OpenFIBSEM will automatically find your Autoscript installation if it installed. On Tescan, please install into the same environment as the Automation API.
- Minimap: Added a minimap widget for collecting tiled images, selecting positions and correlation. Provides an overview of the current stage position and the positions of the selected locations. Also provides an integrated correlation user interface. You can use the minimap to select locations for other applications, such as AutoLamella.  

### Features

- Added a safe_absolute_stage_movement. This function will tilt flat before performing large movements to prevent collions.
- Added cleaning_cross_section and scan_direction to the milling widget user interface.
- Rectangle Patterns now sputter a 'passes' parameter. This allows you to explicitly set the number of passes the beam will scan.
- Adjusted the milling widget to allow for the selection of multiple milling stages. This allows you to move multiple stages together.
- Added automatic logging for alignment data. All alignment data is now logged to a file in the log/crosscorrelation directory. You can change this log directory in the config.
- Added a cryo sputter widget for automated sputtering in cryo conditions.
- Added two way projection between image and stage coordinates. This allows you to click on an image and move the stage to that location, as well as project a stage coordinate to an image coordinate (currently located in fibsem.imaging._tile).
- To move milling stages in the UI, you now need to 'Shift' + 'Left Click' (Was 'Right Click')
- To move the stage vertically (eucentric_move), you now need to 'Alt' + 'Left Click' (Was previously an option in the UI).

### Fixes / Updates

- Fixed an issue where masks were not calculated for alignment.correct_stage_drift.
- Changed the model checkpoint lookup to search the fibsem/segmentation/models directory instead of expecting an absolute path.
- Fixed an issue where coordinate system was flipped when moving using a detection.
- Fixed an issue where milling protocols were being overwritten when setting the milling stages directly. [USER-INTERFACE]
- The milling widget hfw should now update automatically when changing the imaging settings. [USER-INTERFACE]
- The user interface won't try to draw the cross hair if no image is available. [USER-INTERFACE]
- Explicitly converting the last_image to np.uint8 (was np.uint16) [THERMO]
- Explictly settings the manipulator coordinate system when performing movements [THERMO]
- Post milling current now set to 30keV: 150 pA instead of 30keV: UHR Imaging [TESCAN]
- Fixed milling rate conversions, where the milling rate units were not converted correctly [TESCAN]

## 12/07/2023

- Added Documentation
      - Added documentation for the detection and labelling widget
      - Added Instructions for installation using python v-env

- New features
      - Installation and Running .bat scripts
      - Manipulator positions calibration for TESCAN
      - Microscope positions available in the movement widget
      - Added minimap of microscope positions
      - Added a fibsem version number for development tracking
      - Live chat (experimental)
      - Autoliftout utils
      - GIS Widget for cryo-control of gas injection
      - Embedded detection widget

- Fixed bugs
      - fixed issue where parameters were passed incorrectly for milling
      - fixed Eucentric movement where z-direction was flipped

- Updated Functionality / Improved Processes
      - system/model yaml files can now be modified from the system widget
      - demo log paths now in fibsem base directory
      - scan/image rotation now saved to microscope state
      - An option to click to move multiple milling stages together is now available
      - Added a crosshair to the images
      - movement of milling pattern now emits a pyqt signal (backend)
      - Manufacturer / model /serial no info can now be accessed/saved
      - Manipulator UI adaptive based on if manipulator is retracted or inserted
      - Enabled granular hardware control for stage and manipulator (backend), eg: disable rotation only

## 24/05/2023

- Added new features
      - FIB current alignment
      - Manipulator Controls
      - Measurement tools
      - Segment Anything Labelling
      - Added new milling patterns (Bitmap, Annulus)
      - Separated stage pretilt
- Fixed bugs
      - Autolamella example
      - Set microscope stage
      - HFW
      - Milling widget
      - Application file/Presets set on startup
      - Import TESCAN image files
