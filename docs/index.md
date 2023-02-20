# OpenFIBSEM
A universal API for FIBSEM control

## Introduction

Focused Ion Beam Scanning Electron Microscopy (FIBSEM) systems are a powerful tool for imaging and modifying materials at the nanoscale. They consist of a set of common hardware components such as Electron Beam, Ion Beam, Stage, Manipulator, and Gas Injection System. Manufacturers such provide software API to control the hardware. However, each manufacturer has their own proprietary API which is not compatible with other manufacturers. This results in a need for users to learn multiple APIs if they work with different manufacturers or stick to one manufacturer. This also means that applications developed for one manufacturer's hardware may not be portable to another manufacturer's hardware.

To address these issues, OpenFIBSEM provides a common API that can control the hardware across different manufacturers. The aim of OpenFIBSEM is to provide a universal interface that enables users to easily switch between manufacturers without needing to learn new proprietary software. With OpenFIBSEM, users can control FIBSEM systems from different manufacturers with a single, common API. This enables cross-platform applications, making it easier to move applications between different FIBSEM systems without needing to rewrite them for each manufacturer's proprietary API.

![FIBSEM System](https://static.cambridge.org/binary/version/id/urn:cambridge.org:id:binary:20220623180832741-0579:S1551929521001528:S1551929521001528_fig2.png)
A FIBSEM System [Reference](https://www.cambridge.org/core/journals/microscopy-today/article/recent-advances-in-gas-injection-systemfree-cryofib-liftout-transfer-for-cryoelectron-tomography-of-multicellular-organisms-and-tissues/)

OpenFIBSEM aims to provide a core set of functionality, including the ability to control electron and ion beams, the stage, manipulator, and gas injection system. The project aims to expand this functionality over time. In addition, OpenFIBSEM also provides functionality for automation, machine learning, and image processing, enabling users to develop more advanced applications and workflows.

In conclusion, OpenFIBSEM provides a universal API for controlling FIBSEM systems from different manufacturers. By providing a common interface, OpenFIBSEM aims to enable users to easily switch between manufacturers without needing to learn new proprietary software. This also enables cross-platform applications, making it easier to move applications between different FIBSEM systems without needing to rewrite them for each manufacturer's proprietary API. With OpenFIBSEM, users can leverage the power of FIBSEM systems to perform advanced imaging and modification tasks with greater ease and efficiency.

### Supported Manufacturers
Currently, OpenFIBSEM supports two FIBSEM manufacturers: ThermoFisher and Tescan. Support for additional manufacturers will be added in the future. 

For more information on future plans, please see [Roadmap](https://demarcolab.github.io/fibsem/roadmap).

## Getting Started
For information on installation and getting started, please see [Getting Started](https://demarcolab.github.io/fibsem/started).

## Examples
For examples on how to use OpenFIBSEM, please see [Examples](https://demarcolab.github.io/fibsem/exampls).

## API Reference

For a breakdown of the API please see [API Reference](https://demarcolab.github.io/fibsem/reference).