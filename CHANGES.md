## Changes

### 12/07/2023

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


### 24/05/2023

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


