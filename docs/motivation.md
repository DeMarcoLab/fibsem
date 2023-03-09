# Motivation

Here we introduce OpenFIBSEM, a universal API for FIBSEM control. The API aims to provide a single cross-platform interface for controlling FIBSEM systems, and a series of reusable modules and components for microscopy workflows. Due to the diversity of FIBSEM systems and applications, the package focuses on improving the programmability of these systems by focusing on composition, and extendibility. The package implements core functionality such as imaging, movement, milling, manipulator control, provides modules for system calibration, alignment, and image analysis, and re-usable user interface components integrated with napari. The package currently supports ThermoFisher and TESCAN hardware, with support for other manufacturers planned in the future. We will also demonstrate the improved automation driving by OpenFIBSEM by discussing several internally developed FIBSEM applications, such as AutoLamella v2 (automated cryo-lamella preparation), AutoLiftout (automated cryo-liftout), and Salami (automated volume microscopy).

## Background Motivation
In order to provide motivation for this project, we draw a comparison between the current state of the electron microscopy field and the field of computing. In the 1960s, computing was largely limited to academic, government, and high-value commercial applications, and was performed in large shared facilities with expensive equipment operated by experts in the field of computer programming and operation. Programming was done for specific hardware and applications, and users scheduled time and shared resources.

However, the computing industry underwent a dramatic shift, becoming universally used and ubiquitous due to improvements in hardware and software. The development of microprocessors significantly improved computer performance while reducing costs, while the development of operating systems such as Unix enabled programmers to develop programs that could run across different hardware platforms. The use of higher-level programming languages such as C made it easier for programmers to create new programs and opened up programming to a wider range of users. These developments facilitated the automation of manual tasks and drove the computing industry forward.

To make FIBSEM microscopy more accessible and universal, we believe that the field should follow a similar trajectory. Improvements in hardware, including better physics, materials, engineering, and economies of scale, are expected to continue. On the software side, we believe that OpenFIBSEM can provide the foundation for a cross-platform operating system and development ecosystem for FIBSEM systems. Designed with universality, composability, and extendibility in mind, these improvements in programmability of FIBSEM systems will ultimately make it easier to automate existing workflows and develop new applications, thereby driving structural biology research forward.

## Design
The OpenFIBSEM API is designed with three main objectives in mind. First, it aims to be universal, ensuring that users can use the same functionality across different hardware, while enabling cross-platform application development by default. Second, it aims to be composable, allowing users to combine different functionalities to suit their specific requirements. Users should be able to use as much or as little of OpenFIBSEM as they choose, and integrate it with other software tools. Lastly, the API aims to be extendible, enabling users to add new functionalities and interfaces while ensuring compatibility with existing tools. By achieving these three objectives, OpenFIBSEM aims to facilitate the development of advanced FIBSEM applications and workflows in structural biology research.

We aim to achieve the above goals by implementing the following architecture:
![Architecture](/docs/img/roadmap/roadmap_v0.2.png)

The OpenFIBSEM development team has outlined several objectives aimed at improving the API's functionality and accessibility. These objectives include the migration of all existing packages to universal standards, expanding support for a wider range of manufacturers, incorporating additional user modules, and integrating more user interface components. Additionally, the team plans to develop standalone user applications that utilize OpenFIBSEM as their underlying API. These initiatives are expected to improve the overall usability and extendibility of OpenFIBSEM, enabling researchers to develop more advanced FIBSEM applications for structural biology research.

### Universal Standards
The primary interface components are:

### FibsemMicroscope
The FibsemMicroscope is an example of an Abstract Base Class (ABC) in the context of microscopy. It defines a standardized interface that all types of hardware should conform to in order to interact with the package. This enables higher level modules to interact with the microscope without having to worry about the specific type of hardware being used.

In addition, the ABC can add additional functionality to the microscope if available, making it possible to use advanced features of different types of hardware without needing to understand the details of how they work. This implementation provides a powerful abstraction layer that simplifies the process of using microscopy hardware.

We recommend defining at least the most minimal configuration:

```python
class NewMicroscope(FibsemMicroscope):

	def connect(...)
		...
	def disconnect(...)
		...
	def acquire_image(...)
		...
	def move_stage_absolute(...)
		...
	def move_stage_relative(...)
		...
```

### FibsemImage 
The FibsemImage is a common image format that includes metadata specific to Fibsem microscopy. This allows for seamless integration with different types of hardware and ensures that metadata is preserved, and consistent across hardware. The FibsemImage also includes functions that can convert manufacturer-specific formats to the FibsemImage format, which enables users to work with images from different hardware without the need for specialized knowledge.

One of the key benefits of the FibsemImage is that it is used as an input argument for OpenFIBSEM modules. The flexible metadata format can also be extended if required, which provides a high degree of customization to meet specific research needs. Additionally, the storage mechanism for FibsemImage files can be changed if required for larger scale images, (such as in https://www.nature.com/articles/s41592-021-01326-w) and tools will be provided to convert to common formats if needed.


```python
class FibsemImage:
    ...
	def fromNewMicroscope(...) -> FibsemImage:
		... 
	def toNewMicroscope(... ) -> NewMicroscopeImage:
        ...

```

### FibsemCoordinateSystem
The FibsemCoordinateSystem is a unified coordinate system that is designed to ensure that a position or movement within the system is consistent across different types of hardware. The coordinate system is based on a 0, 0, 0 starting point at the electron pole piece and is always in raw format, meaning it is not dependent on the working distance.

Currently, the FibsemCoordinateSystem is not yet enabled as it requires access to a wider range of hardware to test and ensure that it is functioning correctly. However, once it is enabled, it will provide a way to ensure consistency and accuracy when working with Fibsem microscopy data, regardless of the specific type of hardware being used.


---
Just to get it out of the way:

![Hubris](https://imgs.xkcd.com/comics/standards.png)

## Composition

OpenFIBSEM offers modules that are designed to be flexibly composed together to enable users to implement the tools that suit their particular needs. The user interface (UI) is an essential component of FIBSEM systems and applications, but developers face significant challenges in writing the boilerplate code for UIs. This task is time-consuming and increasingly complex as applications become more intricate. To address this challenge, OpenFIBSEM provides reusable UI components, such as those for imaging, movement, and milling, which can be imported into any application. These widgets are integrated with the napari viewer, making them easy to use. Developers can drop these widgets into their application, and they will communicate with the viewer and the microscope. Other modules can invoke these widgets to perform standard UI operations, such as capturing images and updating the display, or enabling double-clicks to move the stage. Since these widgets are composable, developers can choose to use them all, any, or none in their application, in conjunction with the core package. Example applications utilizing these components include FibsemUI, Salami UI, and Autolamella, among others.


### UI Components

| Component | Description | Example |
| --- | --- | --- |
| Viewer | Displays images, patterns and recieves user input | ![Viewer](/docs/img/ui/fibsem_ui.png) |
| Imaging | Captures images from the microscope and displays them in the viewer | ![Image](/docs/img/ui/imaging.png) |
| Movement | Moves the stage and updates the viewer | ![Movement](/docs/img/ui/movement.png) |
| Milling | Draw Patterns and Run Milling | ![Milling](/docs/img/ui/milling.png) |


Eucentric correction is a critical operation that aligns the ion beam and electron beam so that both beams are coincident. To perform this operation, the stage must be centered on a feature in the electron beam and then moved vertically to align the ion beam with the electron beam. OpenFIBSEM offers a cross-platform movement functionality to center the stage by calling microscope.move_eucentric(settings, dy), which moves the stage purely vertically in the ion image plane. This functionality works on both TESCAN and ThermoFisher instruments. To perform this operation, we need the distance to the feature in the ion image, which can be obtained from different sources such as user interface clicks, alignment from a reference image, or feature detection from a machine learning model. By utilizing these components, we can develop multiple eucentric alignment methods, allowing users to implement or develop the method that suits their application and specific conditions, rather than being constrained to predefined workflows.

![ML Eucentric Correction](/docs/img/ml/eucentric_correction/ml_alignment.gif)
ML Based Eucentric Correction Workflow

OpenFIBSEM's open-source nature facilitates frequent updates to its components, methods, tools, and workflows that other researchers might find useful. Contributions from others are welcome, and users can contribute an example workflow by opening a pull request on Github, which will be added to the examples in the repository. For instance, a user may wish to use a proprietary detector and associated API while still benefiting from the core imaging and stage movement functionality offered by OpenFIBSEM.

## Extensibility

OpenFIBSEM is an extensible software package that allows users to implement new functionality and extend the capabilities of FIBSEM systems. Several examples of such extensions are Stable Movement, AutoFocus, AutoGamma, AutoCLAHE, and Charge Neutralisation. Stable Movement corrects for the pre-tilt of the shuttle holder when moving the stage, thus maintaining eucentric position. AutoFocus is a custom routine that can be tuned or extended, while AutoGamma and AutoCLAHE implement multiple image post-processing methods for dealing with excessive charge effects. Charge Neutralisation, on the other hand, develops a procedure for neutralising or building up the charge on a sample.

![Image Post-Processing](/docs/img/gamma/gamma_comparison.png)

OpenFIBSEM supports the interfacing of hardware interactions with a microscope through an implementation of the FibsemMicroscope abstract base class. This class defines the interfaces available for all microscopes to interact with the hardware, and FibsemImage and FibsemCoordinateSystem provide common data structures for user modules to consume and produce. Therefore, extending to a new manufacturer involves implementing each of the base class methods, such as connect, acquire_image, and stage_move_absolute. Although each subclass requires all the base methods to be implemented, in practice, this only means the method needs to be present, not that it does anything. Thus, implementing the core imaging and movement functionality is recommended first, with other methods logging a warning that they are not implemented, or throwing an error until implemented. This approach allows OpenFIBSEM to support a wide range of FIBSEM systems available, each with various subsystems and components.

OpenFIBSEM also enables users to extend it to new modalities, as demonstrated by (Piescope)[CITATION], where integrating additional modalities into the same chamber has significant benefits. For instance, adding a BeamType.LIGHT enumeration to represent a fluorescence microscope could be achieved without changing existing functionality or large-scale rewrites.

```python
class BeamType(Enum):
		ELECTRON = 1
		ION = 2
		LIGHT = 3

class FibsemImage(...):
	…
	@staticmethod
	def fromPIEScope(...) -> ‘FibsemImage’:
		…
	
def acquire_image(microscope, image_settings: ImageSettings) -> FibsemImage:
    if not isinstance(image_settings.beam_type, BeamType):
        return None		
    
    if image_settings.beam_type in [BeamType.ELECTRON, BeamType.ION]:
		return microscope.acquire_image(image_settings)
	
    if image_settings.beam_type is BeamType.Light:
		piescope_image = piescope.acquire_image(image_settings)
        return FibsemImage.fromPIEScope(piescope_image)
```	

 Similarly, different methodologies such as EDS or additional sensors such as a Chamber CCD could also be supported. Because OpenFIBSEM is designed to be extended, adding new functionality does not require significant changes to existing functionality.


 ## User Applications

We have created a range of applications using OpenFIBSEM, which are publicly available on Github. These applications have been intentionally designed to be versatile and adaptable, providing a strong foundation for developing new applications. The applications that we have developed include:

FibsemUI: A versatile User Interface that enables users to access imaging, movement, and milling operations in a seamless and customizable manner.

![FibsemUI](/docs/img/ui/fibsem_ui.png)

Salami: An automated approach to Volume Electron Microscopy that is highly efficient and reliable.

![Salami](/docs/img/ui/salami.png)

AutoLamella v2: An automated cryo-lamella preparation approach that streamlines the process and ensures high-quality results.

![AutoLamellav2](/docs/img/ui/autolamella.png)

We aim to continue extending the functionality of OpenFIBSEM by porting existing applications to this backend and developing new applications that can be used as examples for other users.

