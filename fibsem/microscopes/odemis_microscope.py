import os
import logging
import sys
from copy import deepcopy

import numpy as np

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
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
    PatterningState,
    Point,
    SystemSettings,
    BeamSettings,
    FibsemDetectorSettings,
    MicroscopeState,
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
from odemis.acq.stream import FIBStream, SEMStream
from odemis.util.dataio import open_acquisition

def stage_position_to_odemis_dict(position: FibsemStagePosition) -> dict:
    """Convert a FibsemStagePosition to a dict with the odemis keys"""
    pdict = position.to_dict()
    pdict.pop("name")
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


class OdemisMicroscope(FibsemMicroscope):
    def __init__(self, system_settings: SystemSettings = None):
        self.system: SystemSettings = system_settings

        self.connection: "SEM" = model.getComponent(role="fibsem")

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

        self.user = FibsemUser.from_environment()
        self.experiment = FibsemExperiment()

    def connect_to_microscope(self, ip_address: str, port: int) -> None:
        pass

    def disconnect(self):
        pass

    def acquire_chamber_image(self) -> FibsemImage:
        pass

    def acquire_image(self, image_settings: ImageSettings) -> FibsemImage:
        beam_type = image_settings.beam_type
        channel = beam_type_to_odemis[beam_type]

        # reduced area imaging
        if image_settings.reduced_area is not None:
            scan_mode = "reduced_area"
            scan_mode_value = image_settings.reduced_area.to_dict()
        else:
            scan_mode = "full_frame"
            scan_mode_value = None

        self.connection.set_scan_mode(
            mode=scan_mode, channel=channel, value=scan_mode_value
        )

        # set imaging settings
        # TODO: this is a change in behaviour..., restore the previous conditions or use GrabFrameSettings?
        self.set_imaging_settings(image_settings)

        # acquire image
        image, _md = self.connection.acquire_image(channel=channel)

        # restore to full frame imaging
        if image_settings.reduced_area is not None:
            self.connection.set_scan_mode("full_frame", channel=channel, value=None)

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

        return FibsemImage(image, md)

    def last_image(self, beam_type: BeamType) -> FibsemImage:
        pass

    def autocontrast(self, beam_type: BeamType) -> None:
        self.connection.run_auto_contrast_brightness(beam_type_to_odemis[beam_type])

    def auto_focus(self, beam_type: BeamType) -> None:
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

        logging.warning(f"Unknown key: {key} ({beam_type})")
        return None

    def _set(self, key: str, value: str, beam_type: BeamType = None) -> None:
        # get beam
        if beam_type is not None:
            channel = beam_type_to_odemis[beam_type]

        # beam properties
        if key == "working_distance":
            self.connection.set_working_distance(value, channel)
            try: # TODO: remove linking once testing is done
                if beam_type is BeamType.ELECTRON:
                    self.set("stage_link", True)  # link the specimen stage for electron
            except Exception as e:
                logging.info(f"Failed to link stage: {e}")
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
            self.connection.set_beam_shift(-value.x, value.y, channel)
            logging.info(f"{beam_type.name} shift set to {value}.")
            return
        if key == "stigmation":
            self.connection.set_stigmator(value.x, value.y, channel)
            logging.info(f"{beam_type.name} stigmation set to {value}.")
            return

        if key == "resolution":
            self.connection.set_resolution(value, channel)
            return

        # beam control
        if key == "on":
            self.connection.set_beam_power(value, channel)
            logging.info(f"{beam_type.name} beam turned {'on' if value else 'off'}.")
            return
        if key == "blanked":
            self.connection.beam_blank(
                channel
            ) if value else self.connection.beam_unblank(channel)
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
            values = self.connection.beam_current_info(
                beam_type_to_odemis[beam_type]
            )[
                "range"
            ]  # TODO: we need the list of choices, not the range (this should be an list)
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

    def move_stage_relative(self, position: FibsemStagePosition) -> None:
        pdict = stage_position_to_odemis_dict(position)
        f = self.stage.moveRel(pdict)
        f.result()

    # NOTE: this is an exact copy of the stable_move method from the ThermoMicroscope class, it can be consolidated
    def stable_move(
        self, dx: float, dy: float, beam_type: BeamType, static_wd: bool = False
    ) -> FibsemStagePosition:
        wd = self.get("working_distance", BeamType.ELECTRON)

        scan_rotation = self.get("scan_rotation", beam_type)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0

        # calculate stable movement
        yz_move = self._y_corrected_stage_movement(
            expected_y=dy,
            beam_type=beam_type,
        )
        stage_position = FibsemStagePosition(
            x=dx,
            y=yz_move.y,
            z=yz_move.z,
            r=0,
            t=0,
            coordinate_system="RAW",
        )

        # move stage
        self.move_stage_relative(stage_position)

        # adjust working distance to compensate for stage movement
        if static_wd:
            wd = self.system.electron.eucentric_height

        self.set("working_distance", wd, BeamType.ELECTRON)

        # logging
        logging.debug(
            {
                "msg": "stable_move",
                "dx": dx,
                "dy": dy,
                "beam_type": beam_type.name,
                "static_wd": static_wd,
                "working_distance": wd,
                "scan_rotation": scan_rotation,
                "position": stage_position.to_dict(),
            }
        )

        return stage_position

    def vertical_move(
        self, dy: float, dx: float = 0.0, static_wd: bool = True, use_perspective: bool = True
    ) -> FibsemStagePosition:
        """Move the stage vertically by the specified amount."""

        # get current working distance, to be restored later
        wd = self.get("working_distance", BeamType.ELECTRON)

        # adjust for scan rotation
        scan_rotation = self.get("scan_rotation", BeamType.ION)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0

        # TODO: implement perspective correction
        PERSPECTIVE_CORRECTION = 0.9
        z_move = dy
        if use_perspective:
            z_move = (
                dy
                / np.cos(np.deg2rad(90 - self.system.ion.column_tilt))
                * PERSPECTIVE_CORRECTION
        )  # TODO: MAGIC NUMBER, 90 - fib tilt

        # manually calculate the dx, dy, dz
        theta = self.get_stage_position().t # rad
        dy = z_move * np.sin(theta)
        dz = z_move / np.cos(theta)
        stage_position = FibsemStagePosition(x=dx, y=dy, z=dz)
        self.move_stage_relative(stage_position)


        if static_wd:
            self.set(
                "working_distance",
                self.system.electron.eucentric_height,
                BeamType.ELECTRON,
            )
            self.set("working_distance", self.system.ion.eucentric_height, BeamType.ION)
        else:
            self.set("working_distance", wd, BeamType.ELECTRON)

        # logging
        logging.debug(
            {
                "msg": "vertical_move",
                "dy": dy,
                "dx": dx,
                "static_wd": static_wd,
                "working_distance": wd,
                "scan_rotation": scan_rotation,
                "position": stage_position.to_dict(),
            }
        )

        return self.get_stage_position()

    def _y_corrected_stage_movement(
        self, expected_y: float, beam_type: BeamType
    ) -> FibsemStagePosition:
        """
        Calculate the corrected stage movements based on the beam_type, and then move the stage relatively.

        Args:
            dx (float): distance along the x-axis (image coordinates)
            dy (float): distance along the y-axis (image coordinates)
            beam_type (BeamType): beam type to move in
            static_wd (bool, optional): whether to fix the working distance. Defaults to False.
        """

        # all angles in radians
        stage_tilt_flat_to_electron = np.deg2rad(self.system.electron.column_tilt)
        stage_tilt_flat_to_ion = np.deg2rad(self.system.ion.column_tilt)

        stage_pretilt = np.deg2rad(self.system.stage.shuttle_pre_tilt)

        stage_rotation_flat_to_eb = np.deg2rad(self.system.stage.rotation_reference) % (
            2 * np.pi
        )
        stage_rotation_flat_to_ion = np.deg2rad(self.system.stage.rotation_180) % (
            2 * np.pi
        )

        # current stage position
        current_stage_position = self.get_stage_position()
        stage_rotation = current_stage_position.r % (2 * np.pi)
        stage_tilt = current_stage_position.t

        PRETILT_SIGN = 1.0
        # pretilt angle depends on rotation
        from fibsem import movement

        if movement.rotation_angle_is_smaller(
            stage_rotation, stage_rotation_flat_to_eb, atol=5
        ):
            PRETILT_SIGN = 1.0
        if movement.rotation_angle_is_smaller(
            stage_rotation, stage_rotation_flat_to_ion, atol=5
        ):
            PRETILT_SIGN = -1.0

        # corrected_pretilt_angle = PRETILT_SIGN * stage_tilt_flat_to_electron
        corrected_pretilt_angle = PRETILT_SIGN * (
            stage_pretilt + stage_tilt_flat_to_electron
        )  # electron angle = 0, ion = 52

        # perspective tilt adjustment (difference between perspective view and sample coordinate system)
        if beam_type == BeamType.ELECTRON:
            perspective_tilt_adjustment = -corrected_pretilt_angle
        elif beam_type == BeamType.ION:
            perspective_tilt_adjustment = (
                -corrected_pretilt_angle - stage_tilt_flat_to_ion
            )

        # the amount the sample has to move in the y-axis
        y_sample_move = expected_y / np.cos(
            stage_tilt + perspective_tilt_adjustment
        )

        # the amount the stage has to move in each axis
        y_move = y_sample_move * np.cos(corrected_pretilt_angle)
        z_move = -y_sample_move * np.sin(
            corrected_pretilt_angle
        )  # TODO: investigate this

        return FibsemStagePosition(x=0, y=y_move, z=z_move)

    def project_stable_move(
        self,
        dx: float,
        dy: float,
        beam_type: BeamType,
        base_position: FibsemStagePosition,
    ) -> FibsemStagePosition:
        scan_rotation = self.get("scan_rotation", beam_type)
        if np.isclose(scan_rotation, np.pi):
            dx *= -1.0
            dy *= -1.0

        # stable-move-projection
        point_yz = self._y_corrected_stage_movement(dy, beam_type)
        dy, dz = point_yz.y, point_yz.z

        # calculate the corrected move to reach that point from base-state?
        _new_position = deepcopy(base_position)
        _new_position.x += dx
        _new_position.y += dy
        _new_position.z += dz

        return _new_position

    def safe_absolute_stage_movement(self, position: FibsemStagePosition) -> None:
        self.move_stage_absolute(position)

    def live_imaging(self, beam_type: BeamType) -> None:
        pass

    def consume_image_queue(self):
        pass

    def draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings, path: str):
        pass

    def draw_rectangle(self, pattern_settings: FibsemRectangleSettings):
        pdict = pattern_settings.to_dict()

        pdict["center_x"] = pdict.pop("centre_x")
        pdict["center_y"] = pdict.pop("centre_y")

        # select the correct pattern function
        create_pattern_function = self.connection.create_rectangle
        if pattern_settings.cross_section is CrossSectionPattern.CleaningCrossSection:
            create_pattern_function = self.connection.create_cleaning_cross_section
            self.connection.set_default_application_file("Si-ccs")
        if pattern_settings.cross_section is CrossSectionPattern.RegularCrossSection:
            create_pattern_function = self.connection.create_regular_cross_section
            self.connection.set_default_application_file("Si-multi-pass")

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
        self.milling_channel = mill_settings.milling_channel
        channel = beam_type_to_odemis[self.milling_channel]
        self._default_application_file = mill_settings.application_file
        self.connection.set_default_patterning_beam_type(channel)
        self.connection.set_default_application_file(mill_settings.application_file)
        self.connection.set_patterning_mode(mill_settings.patterning_mode)
        self.connection.clear_patterns()  # clear any existing patterns
        self.set("hfw", mill_settings.hfw, self.milling_channel)
        self.set("current", mill_settings.milling_current, self.milling_channel)
        self.set("voltage", mill_settings.milling_voltage, self.milling_channel)

        logging.debug(
            {"msg": "setup_milling", "mill_settings": mill_settings.to_dict()}
        )

    def start_milling(self):
        self.run_milling(asynch=True)

    def run_milling(
        self,
        milling_current: float = None,
        milling_voltage: float = None,
        asynch: bool = False,
    ):
        try:
            # change to milling current, voltage
            if (
                milling_voltage is not None
                and self.get("voltage", self.milling_channel) != milling_voltage
            ):
                self.set("voltage", milling_voltage, self.milling_channel)
            if (
                milling_current is not None
                and self.get("current", self.milling_channel) != milling_current
            ):
                self.set("current", milling_current, self.milling_channel)
        except Exception as e:
            logging.warning(
                f"Failed to set voltage or current: {e}, voltage={milling_voltage}, current={milling_current}"
            )

        if asynch:
            self.connection.start_milling()
        else:
            self.connection.run_milling()

        logging.debug(
            {
                "msg": "run_milling",
                "milling_current": milling_current,
                "milling_voltage": milling_voltage,
                "asynch": asynch,
            }
        )

    def get_milling_state(self):
        return PatterningState[self.connection.get_patterning_state().upper()]

    def run_milling_drift_corrected(self):
        pass

    def finish_milling(self, imaging_current: float, imaging_voltage: float) -> None:
        self.connection.clear_patterns()
        self.set("current", imaging_current, self.milling_channel)
        self.set("voltage", imaging_voltage, self.milling_channel)
        self.connection.set_patterning_mode("Serial")

        logging.debug(
            {
                "msg": "finish_milling",
                "imaging_current": imaging_current,
                "imaging_voltage": imaging_voltage,
            }
        )

    def stop_milling(self) -> None:
        if self.get_milling_state() is PatterningState.RUNNING:
            self.connection.stop_milling()

    def resume_milling(self) -> None:
        self.connection.resume_milling()

    def pause_milling(self) -> None:
        self.connection.pause_milling()

    def estimate_milling_time(self, patterns: list = None) -> float:
        return self.connection.estimate_milling_time()
