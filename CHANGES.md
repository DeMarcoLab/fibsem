# Changes

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
