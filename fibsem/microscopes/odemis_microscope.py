import logging
import os
import sys
from copy import deepcopy
from typing import Optional, TYPE_CHECKING
import numpy as np
from psygnal import Signal

from fibsem.microscope import FibsemMicroscope, ThermoMicroscope
from fibsem.structures import (
    BeamSettings,
    BeamType,
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemDetectorSettings,
    FibsemExperiment,
    FibsemImage,
    FibsemImageMetadata,
    FibsemLineSettings,
    FibsemManipulatorPosition,
    FibsemMillingSettings,
    FibsemRectangleSettings,
    FibsemStagePosition,
    FibsemUser,
    ImageSettings,
    FibsemRectangle,
    MicroscopeState,
    MillingState,
    Point,
    SystemSettings,
    ACTIVE_MILLING_STATES,
)


def add_odemis_path():
    """Add the odemis path to the python path"""

    def parse_config(path) -> dict:
        """Parse the odemis config file and return a dict with the config values"""

        with open(path) as f:
            config = f.read()

        config = config.split("\n")
        config = [line.split("=") for line in config]
        config = {
            line[0]: line[1].replace('"', "") for line in config if len(line) == 2
        }
        return config

    odemis_path = "/etc/odemis.conf"
    config = parse_config(odemis_path)
    sys.path.append(f"{config['DEVPATH']}/odemis/src")  # dev version
    sys.path.append("/usr/lib/python3/dist-packages")  # release version + pyro4


add_odemis_path()

from odemis import model
from odemis.util.dataio import open_acquisition


if TYPE_CHECKING:
    from odemis.driver.autoscript_client import SEM as OdemisAutoscriptClient


def stage_position_to_odemis_dict(position: FibsemStagePosition) -> dict:
    """Convert a FibsemStagePosition to a dict with the odemis keys"""
    pdict = position.to_dict()
    pdict.pop("name")
    pdict.pop("coordinate_system") # no longer used in odemis
    pdict["rz"] = pdict.pop("r")
    pdict["rx"] = pdict.pop("t")

    # if any values are None, remove them
    pdict = {k: v for k, v in pdict.items() if v is not None}

    return pdict


def odemis_dict_to_stage_position(pdict: dict) -> FibsemStagePosition:
    """Convert a dict with the odemis keys to a FibsemStagePosition"""

    pdict = deepcopy(pdict)

    pdict["r"] = pdict.pop("rz")
    pdict["t"] = pdict.pop("rx")
    pdict["coordinate_system"] = pdict.get("coordinate_system", "RAW")
    return FibsemStagePosition.from_dict(pdict)


def beam_settings_from_odemis_dict(channel: str, md: dict, wd: float) -> BeamSettings:
    c2b = {"electron": BeamType.ELECTRON, "ion": BeamType.ION}

    shift = md["shift"][0]
    stigmator = md["stigmator"][0]

    return BeamSettings(
        beam_type=c2b[channel],
        working_distance=wd,
        beam_current=md["beamCurrent"][0],
        dwell_time=md["dwellTime"][0],
        voltage=md["accelVoltage"][0],
        hfw=md["horizontalFoV"][0],
        resolution=md["resolution"][0],
        scan_rotation=md["rotation"][0],
        shift=Point(shift[0], shift[1]),
        stigmation=Point(stigmator[0], stigmator[1]),
    )


def detector_settings_from_odemis_dict(md: dict) -> FibsemDetectorSettings:
    return FibsemDetectorSettings(
        type=md["type"][0],
        mode=md["mode"][0],
        brightness=md["brightness"][0],
        contrast=md["contrast"][0],
    )


def odemis_md_to_microscope_state(md) -> MicroscopeState:
    # stage position
    stage_md = md["Stage"]["position"][0]
    stage_position = FibsemStagePosition.from_odemis_dict(stage_md)

    # electron beam
    ebs = BeamSettings.from_odemis_dict(
        channel="electron",
        md=md["Electron-Beam"],
        wd=md["Electron-Focus"]["position"][0]["z"],
    )

    # ion beam
    ibs = BeamSettings.from_odemis_dict(
        channel="ion", md=md["Ion-Beam"], wd=md["Ion-Focus"]["position"][0]["z"]
    )

    # electron detector
    eds = FibsemDetectorSettings.from_odemis_dict(md["Electron-Detector"])

    # ion detector
    ids = FibsemDetectorSettings.from_odemis_dict(md["Ion-Detector"])

    ms = MicroscopeState(
        stage_position=stage_position,
        electron_beam=ebs,
        electron_detector=eds,
        ion_beam=ibs,
        ion_detector=ids,
    )

    return ms

def from_odemis_image(image: model.DataArray, path: str = None) -> FibsemImage:
    md = image.metadata
    ms = MicroscopeState.from_odemis_dict(md[model.MD_EXTRA_SETTINGS])

    # image settings
    pixel_size = Point.from_list(md[model.MD_PIXEL_SIZE])

    # TODO: this should be acq_type, but it's not saved in the metadata yet...
    d2b = {"SEM": BeamType.ELECTRON, "FIB": BeamType.ION}

    if d2b[md[model.MD_DESCRIPTION]] is BeamType.ELECTRON:
        sys_state = ms.electron_beam
    if d2b[md[model.MD_DESCRIPTION]] is BeamType.ION:
        sys_state = ms.ion_beam

    filename = None
    if path is not None:
        path = os.path.dirname(path)
        filename = os.path.basename(path)

    image_settings = ImageSettings(
        resolution = [image.shape[1], image.shape[0]],
        dwell_time=sys_state.dwell_time,
        hfw=sys_state.hfw,
        beam_type=sys_state.beam_type,
        path = path,
        filename=filename,
        save = True,
        autocontrast=False,
        autogamma=False,
    )

    image_md = FibsemImageMetadata(
        image_settings=image_settings,
        pixel_size=pixel_size,
        microscope_state=ms,
        # TODO: the rest of the metadata is not saved in the odemis metadata
    )

    # get the data
    da = image.getData() if isinstance(image, model.DataArrayShadow) else image
    
    return FibsemImage(data=da, metadata=image_md)

def load_odemis_image(path: str) -> FibsemImage:
    """Load an odemis image from a file and convert it to a FibsemImage"""
    acq = open_acquisition(path)
    image: FibsemImage = FibsemImage.from_odemis(acq[0], path=path)
    return image

# add as class methods
FibsemStagePosition.to_odemis_dict = stage_position_to_odemis_dict
FibsemStagePosition.from_odemis_dict = odemis_dict_to_stage_position
BeamSettings.from_odemis_dict = beam_settings_from_odemis_dict
FibsemDetectorSettings.from_odemis_dict = detector_settings_from_odemis_dict
MicroscopeState.from_odemis_dict = odemis_md_to_microscope_state
FibsemImage.from_odemis = from_odemis_image
FibsemImage.load_odemis_image = load_odemis_image

beam_type_to_odemis = {
    BeamType.ELECTRON: "electron",
    BeamType.ION: "ion",
}

# TODO: load default system settings?

class OdemisMicroscope(FibsemMicroscope):
    milling_progress_signal = Signal(dict)
    _last_imaging_settings: ImageSettings

    def __init__(self, system_settings: SystemSettings = None):
        self.system: SystemSettings = system_settings

        self.connection: OdemisAutoscriptClient = model.getComponent(role="fibsem")

        # stage
        self.stage: model.Actuator = model.getComponent(role="stage-bare")

        logging.info("OdemisMicroscope initialized")

        # system information # TODO: split this version info properly
        software_version = self.connection.get_software_version()
        hardware_version = self.connection.get_hardware_version()
        self.system.info.model = hardware_version
        self.system.info.serial_number = hardware_version
        self.system.info.hardware_version = software_version
        self.system.info.software_version = software_version
        info = self.system.info
        logging.info(
            f"Microscope client connected to model {info.model} with serial number {info.serial_number} and software version {info.software_version}."
        )

        # internal parameters
        self.stage_is_compustage = False
        self.milling_channel: BeamType = BeamType.ION
        self._default_application_file: str = "Si"
        self._last_imaging_settings: ImageSettings = ImageSettings()

        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperiment()

    def connect_to_microscope(self, ip_address: str, port: int) -> None:
        pass

    def disconnect(self):
        pass

    def get_orientation(self, orientation: str) -> str:
        """Get the current orientation of the microscope."""
        return ThermoMicroscope.get_orientation(self, orientation)

    def get_stage_orientation(self, stage_position: Optional[FibsemStagePosition] = None) -> str:
        """Get the stage position for the specified orientation."""
        return ThermoMicroscope.get_stage_orientation(self, stage_position)

    def move_flat_to_beam(self, beam_type: BeamType, _safe: bool = True) -> None:
        # new style
        omap = {BeamType.ELECTRON: "SEM", BeamType.ION: "FIB"}
        pos = self.get_orientation(omap[beam_type])
        rotation, tilt = pos.r, pos.t
        stage_orientation = self.get_stage_orientation()

        # updated safe rotation move
        logging.info(f"moving flat to {beam_type.name}")
        stage_position = FibsemStagePosition(r=rotation, t=tilt, coordinate_system="Raw")

        # imitate compucentric movements
        if (stage_orientation in ["SEM", "MILLING"] and beam_type == BeamType.ION) or \
           (stage_orientation == "FIB" and beam_type == BeamType.ELECTRON):

            current_stage_position = self.get_stage_position()
            stage_position.x = -current_stage_position.x
            stage_position.y = -current_stage_position.y
            stage_position.z =  current_stage_position.z

        logging.debug({"msg": "move_flat_to_beam", "stage_position": stage_position.to_dict(), "beam_type": beam_type.name})

        if _safe:
            self.safe_absolute_stage_movement(stage_position)
        else:
            self.move_stage_absolute(stage_position)

    def set_channel(self, channel: BeamType):
        """Set the active channels for the microscope."""
        self.connection.set_active_view(channel.value)
        self.connection.set_active_device(channel.value)

    def acquire_chamber_image(self) -> FibsemImage:
        pass

    def acquire_image(self, image_settings: ImageSettings) -> FibsemImage:
        # TODO: migrate to updated api that allows acquiring without setting the imaging settings first
        beam_type = image_settings.beam_type
        channel = beam_type_to_odemis[beam_type]

        # reduced area imaging
        if image_settings.reduced_area is not None:
            reduced_area = image_settings.reduced_area
            self.connection.set_reduced_area_scan_mode(channel=channel, 
                                                       left=reduced_area.left,
                                                       top=reduced_area.top,
                                                       width=reduced_area.width,
                                                       height=reduced_area.height)
        else:
            self.connection.set_full_frame_scan_mode(channel=channel)

        # set imaging settings
        # TODO: this is a change in behaviour..., restore the previous conditions or use GrabFrameSettings?
        # This is the source of the error with square resolutions. 
        # can't set square resolution, but can acquire an image with square
        frame_settings = None
        tmp_resolution = None
        resolution = image_settings.resolution
        if resolution[0] == resolution[1]:
            # can't set square resolution directly
            frame_settings = {"resolution": f"{resolution[0]}x{resolution[1]}"}
            tmp_resolution = resolution
            image_settings.resolution = self.get_resolution(beam_type=beam_type)
        self.set_imaging_settings(image_settings)

        # acquire image
        image, _md = self.connection.acquire_image(channel=channel, frame_settings=frame_settings)

        # restore to full frame imaging
        if image_settings.reduced_area is not None:
            self.connection.set_full_frame_scan_mode(channel=channel)

        # restore the previous resolution
        if tmp_resolution is not None:
            image_settings.resolution = tmp_resolution

        # create metadata
        # TODO: retrieve the full image metadata from image md, rather than reconstruct
        pixel_size = image_settings.hfw / image_settings.resolution[0]
        md = FibsemImageMetadata(
            image_settings=image_settings,
            pixel_size=Point(pixel_size, pixel_size),
            microscope_state=self.get_microscope_state(
                beam_type=image_settings.beam_type
            ),
            user=self.user,
            experiment=self.experiment,
            system=self.system,
        )

        # store last imaging settings
        self._last_imaging_settings = image_settings

        return FibsemImage(image, md)

    def last_image(self, beam_type: BeamType) -> FibsemImage:
        pass

    def autocontrast(self, beam_type: BeamType, reduced_area: FibsemRectangle = None) -> None:
        channel = beam_type_to_odemis[beam_type]
        if reduced_area is not None:
            self.connection.set_reduced_area_scan_mode(channel, **reduced_area.to_dict())
        self.connection.run_auto_contrast_brightness(channel=channel)
        if reduced_area is not None:
            self.connection.set_full_frame_scan_mode(channel)

    def auto_focus(self, beam_type: BeamType, reduced_area: Optional[FibsemRectangle] = None) -> None:        
        self.connection.run_auto_focus(beam_type_to_odemis[beam_type])

    def beam_shift(self, dx: float, dy: float, beam_type: BeamType) -> None:
        self.connection.move_beam_shift(dx, dy, beam_type_to_odemis[beam_type])

    def _get(self, key: str, beam_type: BeamType = None) -> str:
        if beam_type is not None:
            channel = beam_type_to_odemis[beam_type]

        # beam properties
        if key == "on":
            return self.connection.get_beam_is_on(channel)
        if key == "blanked":
            return self.connection.beam_is_blanked(channel)
        if key == "working_distance":
            return self.connection.get_working_distance(channel)

        if key == "current":
            return self.connection.get_beam_current(channel)
        if key == "voltage":
            return self.connection.get_high_voltage(channel)
        if key == "hfw":
            return self.connection.get_field_of_view(channel)
        if key == "dwell_time":
            return self.connection.get_dwell_time(channel)
        if key == "scan_rotation":
            return self.connection.get_scan_rotation(channel)
        if key == "voltage_limits":
            voltage_info = self.connection.high_voltage_info(channel)
            return [voltage_info["range"][0], voltage_info["range"][1]]
        if key == "voltage_controllable":
            return True
        if key == "shift":  # beam shift
            beam_shift = self.connection.get_beam_shift(channel)
            return Point(beam_shift[0], beam_shift[1])
        if key == "stigmation":
            stigmation = self.connection.get_stigmator(channel)
            return Point(stigmation[0], stigmation[1])
        if key == "resolution":
            width, height = self.connection.get_resolution(channel)
            return [width, height]

        # system properties
        if key == "eucentric_height":
            if beam_type is BeamType.ELECTRON:
                return self.system.electron.eucentric_height
            elif beam_type is BeamType.ION:
                return self.system.ion.eucentric_height
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")

        if key == "column_tilt":
            if beam_type is BeamType.ELECTRON:
                return self.system.electron.column_tilt
            elif beam_type is BeamType.ION:
                return self.system.ion.column_tilt
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")

        # ion beam properties
        if key == "plasma":
            if beam_type is BeamType.ION:
                return self.system.ion.plasma
            else:
                return False

        if key == "plasma_gas":
            if beam_type is BeamType.ION and self.system.ion.plasma:
                raise NotImplementedError()
            else:
                return None

        # stage properties
        if key == "stage_position":
            pdict = self.stage.position.value
            return FibsemStagePosition.from_odemis_dict(pdict)

        if key == "stage_homed":
            return self.connection.is_homed()
        if key == "stage_linked":
            return self.connection.is_linked()

        # chamber properties
        if key == "chamber_state":
            return self.connection.vacuum.chamber_state

        if key == "chamber_pressure":
            return self.connection.vacuum.chamber_pressure.value

        # detector mode and type
        if key == "detector_type":
            return self.connection.get_detector_type(channel)
        if key == "detector_mode":
            return self.connection.get_detector_mode(channel)
        if key == "detector_brightness":
            return self.connection.get_brightness(channel)
        if key == "detector_contrast":
            return self.connection.get_contrast(channel)

        # manipulator properties
        if key == "manipulator_position":
            raise NotImplementedError()
        if key == "manipulator_state":
            raise NotImplementedError()

        # manufacturer properties
        if key == "manufacturer":
            return self.system.info.manufacturer
        if key == "model":
            return self.system.info.model
        if key == "serial_number":
            return self.system.info.serial_number
        if key == "software_version":
            return self.system.info.software_version
        if key == "hardware_version":
            return self.system.info.hardware_version

        if key in ["preset"]:
            return None

        logging.warning(f"Unknown key: {key} ({beam_type})")
        return None

    def _set(self, key: str, value: str, beam_type: BeamType = None) -> None:
        # get beam
        if beam_type is not None:
            channel = beam_type_to_odemis[beam_type]

        # beam properties
        if key == "working_distance":
            self.connection.set_working_distance(value, channel)
            logging.info(f"{beam_type.name} working distance set to {value} m.")
            return
        if key == "current":
            self.connection.set_beam_current(value, channel)
            logging.info(f"{beam_type.name} current set to {value} A.")
            return
        if key == "voltage":
            self.connection.set_high_voltage(value, channel)
            logging.info(f"{beam_type.name} voltage set to {value} V.")
            return
        if key == "hfw":
            self.connection.set_field_of_view(value, channel)
            logging.info(f"{beam_type.name} HFW set to {value} m.")
            return
        if key == "dwell_time":
            self.connection.set_dwell_time(value, channel)
            logging.info(f"{beam_type.name} dwell time set to {value} s.")
            return
        if key == "scan_rotation":
            self.connection.set_scan_rotation(value, channel)
            logging.info(f"{beam_type.name} scan rotation set to {value} radians.")
            return
        if key == "shift":
            self.connection.set_beam_shift(value.x, value.y, channel)
            logging.info(f"{beam_type.name} shift set to {value}.")
            return
        if key == "stigmation":
            self.connection.set_stigmator(value.x, value.y, channel)
            logging.info(f"{beam_type.name} stigmation set to {value}.")
            return

        if key == "resolution":
            self.connection.set_resolution(value, channel)
            return

        # patterning
        if key == "patterning_mode":
            if value in ["Serial", "Parallel"]:
                self.connection.set_patterning_mode(value)
                logging.info(f"Patterning mode set to {value}.")
                return
        if key == "application_file":
            self.connection.set_default_application_file(value)
            logging.info(f"Default application file set to {value}.")
            return
        if key == "default_patterning_beam_type":
            channel = beam_type_to_odemis[value]
            self.connection.set_default_patterning_beam_type(channel)
            logging.info(f"Patterning beam type set to {value} - {channel} .")
            return


        # beam control
        if key == "on":
            self.connection.set_beam_power(value, channel)
            logging.info(f"{beam_type.name} beam turned {'on' if value else 'off'}.")
            return
        if key == "blanked":
            self.connection.blank_beam(
                channel
            ) if value else self.connection.unblank_beam(channel)
            logging.info(
                f"{beam_type.name} beam {'blanked' if value else 'unblanked'}."
            )
            return

        # detector properties
        if key == "detector_mode":
            if value in self.get_available_values("detector_mode", beam_type):
                self.connection.set_detector_mode(value, channel)
                logging.info(f"Detector mode set to {value}.")
            else:
                logging.warning(f"Detector mode {value} not available.")
            return
        if key == "detector_type":
            if value in self.get_available_values("detector_type", beam_type):
                self.connection.set_detector_type(value, channel)
                logging.info(f"Detector type set to {value}.")
            else:
                logging.warning(f"Detector type {value} not available.")
            return
        if key == "detector_brightness":
            if 0 < value <= 1:
                self.connection.set_brightness(value, channel)
                logging.info(f"Detector brightness set to {value}.")
            else:
                logging.warning(
                    f"Detector brightness {value} not available, must be between 0 and 1."
                )
            return
        if key == "detector_contrast":
            if 0 < value <= 1:
                self.connection.set_contrast(value, channel)
                logging.info(f"Detector contrast set to {value}.")
            else:
                logging.warning(
                    f"Detector contrast {value} not available, mut be between 0 and 1."
                )
            return

        if key == "spot_mode":
            self.connection.set_spot_scan_mode(channel=channel, x=value.x, y=value.y)
            return

        if key == "full_frame":
            self.connection.set_full_frame_scan_mode(channel)
            return
        # system properties
        if key == "beam_enabled":
            if beam_type is BeamType.ELECTRON:
                self.system.electron.beam.enabled = value
                return
            elif beam_type is BeamType.ION:
                self.system.ion.beam.enabled = value
                return
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")
            return

        if key == "eucentric_height":
            if beam_type is BeamType.ELECTRON:
                self.system.electron.eucentric_height = value
                return
            elif beam_type is BeamType.ION:
                self.system.ion.eucentric_height = value
                return
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")

        if key == "column_tilt":
            if beam_type is BeamType.ELECTRON:
                self.system.electron.column_tilt = value
                return
            elif beam_type is BeamType.ION:
                self.system.ion.column_tilt = value
                return
            else:
                raise ValueError(f"Unknown beam type: {beam_type} for {key}")

        # ion beam properties
        if key == "plasma":
            if beam_type is BeamType.ION:
                self.system.ion.plasma = value
                return

        # ion beam properties
        if beam_type is BeamType.ION:
            if key == "plasma_gas":
                if not self.system.ion.plasma:
                    logging.debug("Plasma gas cannot be set on this microscope.")
                    return
                if not self.check_available_values("plasma_gas", [value], beam_type):
                    logging.warning(
                        f"Plasma gas {value} not available. Available values: {self.get_available_values('plasma_gas', beam_type)}"
                    )

                logging.info(
                    f"Setting plasma gas to {value}... this may take some time..."
                )
                raise NotImplementedError()
                logging.info(f"Plasma gas set to {value}.")

                return

        # stage properties
        if key == "stage_home":
            logging.info("Homing stage...")
            self.connection.home_stage()
            logging.info("Stage homed.")
            return

        if key == "stage_link":
            if self.stage_is_compustage:
                logging.debug("Compustage does not support linking.")
                return

            logging.info("Linking stage...")
            self.connection.link(value)
            logging.info(f"Stage {'linked' if value else 'unlinked'}.")
            return

        # chamber properties
        if key == "pump_chamber":
            if value:
                logging.info("Pumping chamber...")
                self.connection.pump()
                logging.info("Chamber pumped.")
                return
            else:
                logging.warning(f"Invalid value for pump_chamber: {value}.")
                return

        if key == "vent_chamber":
            if value:
                logging.info("Venting chamber...")
                self.connection.vent()
                logging.info("Chamber vented.")
                return
            else:
                logging.warning(f"Invalid value for vent_chamber: {value}.")
                return

        if key == "active_view":
            self.connection.set_active_view(value.value) # value == BeamType
            return
        if key == "active_device":
            self.connection.set_active_device(value.value) # value == BeamType
            return

        # known keys that are not implemented
        if key in ["preset"]:
            return

        logging.warning(f"Unknown key: {key} ({beam_type})")

        return

    def _get_saved_manipulator_position(self, name: str) -> FibsemManipulatorPosition:
        pass

    def get_available_values(self, key: str, beam_type: BeamType = None) -> list:
        values = []
        if key == "application_file":
            values = self.connection.get_available_application_files()
        if key == "scan_direction":
            values = ["TopToBottom", "BottomToTop", "LeftToRight", "RightToLeft"]
        if key == "detector_type":
            values = self.connection.detector_type_info(beam_type_to_odemis[beam_type])[
                "choices"
            ]
        if key == "detector_mode":
            values = self.connection.detector_mode_info(beam_type_to_odemis[beam_type])[
                "choices"
            ]
        if key == "current":
            values = self.connection.beam_current_info(beam_type_to_odemis[beam_type])["choices"]
            # xenon
            # values = [1.0e-12, 3.0e-12, 10e-12, 30e-12, 0.1e-9, 0.3e-9, 1e-9, 4e-9, 15e-9, 60e-9]
            # argon
            # values = [1.0e-12, 6.0e-12, 20e-12, 60e-12, 0.2e-9, 0.74e-9, 2.0e-9, 7.4e-9, 28.0e-9, 120.0e-9]
        if key == "plasma_gas":
            values = ["Argon", "Oxygen", "Xenon"]

        logging.debug({"msg": "get_available_values", "key": key, "values": values})

        return values

    def check_available_values(self, key: str) -> list:
        pass

    def insert_manipulator(self) -> None:
        pass

    def move_manipulator_absolute(self, position: FibsemManipulatorPosition) -> None:
        pass

    def move_manipulator_relative(self, position: FibsemManipulatorPosition) -> None:
        pass

    def move_manipulator_corrected(self, position: FibsemManipulatorPosition) -> None:
        pass

    def move_manipulator_to_position_offset(
        self, offset: FibsemManipulatorPosition, name: str
    ) -> None:
        pass

    def retract_manipulator(self) -> None:
        pass

    def move_stage_absolute(self, position: FibsemStagePosition) -> None:
        pdict = stage_position_to_odemis_dict(position)
        f = self.stage.moveAbs(pdict)
        f.result()
        # TODO: implement compucentric rotation

    def move_stage_relative(self, position: FibsemStagePosition) -> None:
        pdict = stage_position_to_odemis_dict(position)
        f = self.stage.moveRel(pdict)
        f.result()

    def stable_move(
        self, dx: float, dy: float, beam_type: BeamType, static_wd: bool = False
    ) -> FibsemStagePosition:
        return ThermoMicroscope.stable_move(self, dx=dx, dy=dy, beam_type=beam_type, static_wd=static_wd)

    def vertical_move(
        self, dy: float, dx: float = 0.0, static_wd: bool = True
    ) -> FibsemStagePosition:
        """Move the stage vertically by the specified amount."""
        return ThermoMicroscope.vertical_move(self, dy=dy, dx=dx, static_wd=static_wd)

    def _y_corrected_stage_movement(
        self, expected_y: float, beam_type: BeamType
    ) -> FibsemStagePosition:
        return ThermoMicroscope._y_corrected_stage_movement(self, expected_y, beam_type)

    def project_stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        base_position: FibsemStagePosition,
    ) -> FibsemStagePosition:
        return ThermoMicroscope.project_stable_move(self, 
                                                    dx=dx, dy=dy, 
                                                    beam_type=beam_type, 
                                                    base_position=base_position)

    def _safe_rotation_movement(self, stage_position: FibsemStagePosition) -> None:
        return ThermoMicroscope._safe_rotation_movement(self, stage_position)

    def safe_absolute_stage_movement(self, position: FibsemStagePosition) -> None:
        return ThermoMicroscope.safe_absolute_stage_movement(self, position)

    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings, path: str):
        pass

    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings):
        pdict = pattern_settings.to_dict()

        pdict["center_x"] = pdict.pop("centre_x")
        pdict["center_y"] = pdict.pop("centre_y")

        # select the correct pattern function
        create_pattern_function = self.connection.create_rectangle
        self.connection.set_default_application_file("Si")
        if pattern_settings.cross_section is CrossSectionPattern.CleaningCrossSection:
            create_pattern_function = self.connection.create_cleaning_cross_section
            self.connection.set_default_application_file("Si-ccs")
        if pattern_settings.cross_section is CrossSectionPattern.RegularCrossSection:
            create_pattern_function = self.connection.create_regular_cross_section
            self.connection.set_default_application_file("Si-multipass")

        # create the pattern (draw)
        pinfo = create_pattern_function(pdict)

        # restore the default application file
        self.connection.set_default_application_file(self._default_application_file)

        logging.debug(
            {
                "msg": "draw_rectangle",
                "pattern_settings": pattern_settings.to_dict(),
                "pinfo": pinfo,
            }
        )

    def draw_line(self, pattern_settings: FibsemLineSettings):
        pdict = pattern_settings.to_dict()

        self.connection.set_default_application_file("Si")

        pinfo = self.connection.create_line(pdict)

        self.connection.set_default_application_file(self._default_application_file)

        logging.debug(
            {
                "msg": "draw_line",
                "pattern_settings": pattern_settings.to_dict(),
                "pinfo": pinfo,
            }
        )

    def draw_circle(self, pattern_settings: FibsemCircleSettings):
        pdict = pattern_settings.to_dict()
        pdict["outer_diameter"] = 2 * pattern_settings.radius
        pdict["inner_diameter"] = 0
        pdict["center_x"] = pattern_settings.centre_x
        pdict["center_y"] = pattern_settings.centre_y

        self.connection.set_default_application_file("Si")

        pinfo = self.connection.create_circle(pdict)

        self.connection.set_default_application_file(self._default_application_file)

        logging.debug(
            {
                "msg": "draw_circle",
                "pattern_settings": pattern_settings.to_dict(),
                "pinfo": pinfo,
            }
        )

    def setup_sputter(self):
        pass

    def draw_sputter_pattern(self):
        pass

    def run_sputter(self, *args, **kwargs):
        pass

    def finish_sputter(self):
        pass

    def cryo_deposition_v2(self):
        pass

    def setup_milling(self, mill_settings: FibsemMillingSettings):
        self._default_application_file = mill_settings.application_file
        self.milling_channel = mill_settings.milling_channel
        self.set_milling_settings(mill_settings)
        self.clear_patterns()

        logging.debug(
            {"msg": "setup_milling", "mill_settings": mill_settings.to_dict()}
        )

    def run_milling(self, milling_current: float, milling_voltage: float, asynch: bool = False):
        ThermoMicroscope.run_milling(self, milling_current, milling_voltage, asynch)

    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        ThermoMicroscope.finish_milling(self, imaging_current, imaging_voltage)

    def clear_patterns(self) -> None:
        self.connection.clear_patterns()

    def get_milling_state(self):
        return MillingState[self.connection.get_patterning_state().upper()]

    def start_milling(self) -> None:
        """Start the milling process."""
        if self.get_milling_state() is MillingState.IDLE:
            self.connection.start_milling()
            logging.info("Starting milling...")

    def stop_milling(self) -> None:
        """Stop the milling process."""
        if self.get_milling_state() in ACTIVE_MILLING_STATES:
            logging.info("Stopping milling...")
            self.connection.stop_milling()
            logging.info("Milling stopped.")

    def pause_milling(self) -> None:
        """Pause the milling process."""
        if self.get_milling_state() == MillingState.RUNNING:
            logging.info("Pausing milling...")
            self.connection.pause_milling()
            logging.info("Milling paused.")

    def resume_milling(self) -> None:
        """Resume the milling process."""
        if self.get_milling_state() == MillingState.PAUSED:
            logging.info("Resuming milling...")
            self.connection.resume_milling()
            logging.info("Milling resumed.")   
    
    def estimate_milling_time(self) -> float:
        return self.connection.estimate_milling_time()
