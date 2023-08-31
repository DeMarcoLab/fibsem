# Milling

The milling section allows the user to setup and control the milling process using the FIB. This includes setting up the milling patterns and specific milling settings.

![milling_tab](../img/user_guide/milling/milling_tab.png)

The milling process is done in milling stages. Each milling stage comprises of a pattern and the milling settings required for this pattern. The milling stages are then executed in the order of creation. A milling stage can be added by clicking the 'Add' button. An ion beam image must be taken first before a milling stage can be added. The stages are named in order of creation. A stage can be removed by pressing the remove button. A specific milling stage can be selected by choosing it from the milling stage drop down button.

## Milling Settings

For each milling stage, there are milling settings associated with it, along with the pattern to be milled. The milling settings include: milling rate (specified in mm<sup>3</sup> / A / s), the dwell time (in microseconds), the spot size (in micron), spacing and current (A) or Preset in the case of TESCAN system. In the case of ThermoFisher systems, an application file is available. 

## Milling Patterns

The pattern specifies the kind of pattern to be milled. These include simple shapes such as rectangles, circles and lines. However, OpenFIBSEM also includes a host of complex shapes ideal for lamella preparation. Each pattern has its own unqiue set of parameters that can be set. The full list of patterns is includes:
* Rectangle
* Line
* Circle
* Trench
* Horseshoe
* Undercut
* Fiducial
* Spot weld
* Micro Expansion
* Waffle Notch
* Clover
* Triforce
* Annulus 
* Bitmap (ThermoFisher Only)

 By default, when changes are made to the pattern, the pattern displayed in the viewer is updated live. However, this can be disabled if necessary by unchecking the 'Live Update' checkbox.

A pattern's position can be chosen relative to it's location on the ion image. This can be manually moved by entering coordinates in the centre x and centre y field. The coordinates of (0,0) is the centre of the image.

Alternatively, the pattern can be moved anywhere within the image by pressing the *Shift* key and left-clicking on the image. If the pattern is out of bounds, it will return an out-of-bounds error and the pattern will remain where it was.

## Multiple Stages

Multiple stages can be setup at once. In this manner, once multiple stages are ready, clicking "Run Milling" will run all the stages sequentially without any more manual input. This avoids the repetitive nature of setting up and running stages individually. 



