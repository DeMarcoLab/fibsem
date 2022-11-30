from abc import ABC, abstractmethod
import fibsem.utils as utils
from pathlib import Path
import os
import logging
from autoscript_sdb_microscope_client import SdbMicroscopeClient

class FibsemMicroscope(ABC):
    @abstractmethod
    def setup_session(self, host: str, port: int):
        pass

    @abstractmethod
    def disconnect(self):
        pass


class ThermoMicroscope(FibsemMicroscope):
    def __init__(self, session_path: Path = None, config_path: Path = None, protocol_path: Path = None, setup_logging: bool = True):
        self.setup_session()


    @staticmethod
    def setup_session(
        self,
        session_path: Path = None,
        config_path: Path = None,
        protocol_path: Path = None,
        setup_logging: bool = True,
    ):
        self.settings = utils.load_settings_from_config(config_path, protocol_path)

        # create session directories
        session = f'{self.settings.protocol["name"]}_{utils.current_timestamp()}'
        if protocol_path is None:
            protocol_path = os.getcwd()

        # configure paths
        if session_path is None:
            session_path = os.path.join(os.path.dirname(protocol_path), session)
        os.makedirs(session_path, exist_ok=True)

        # configure logging
        if setup_logging:
            utils.configure_logging(session_path)

        # connect to microscope
        self.__connect_to_microscope__(ip_address=self.settings.system.ip_address)

        # image_setttings
        self.settings.image.save_path = session_path

        logging.info(f"Finished setup for session: {session}")        

    def disconnect(self):
        pass

    def __connect_to_microscope__(self, ip_address: str, port: int):
        """Connect to the FIBSEM microscope."""
        try:
            # TODO: get the port
            logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
            self.microscope = SdbMicroscopeClient()
            self.microscope.connect(host=ip_address, port=port)
            logging.info(f"Microscope client connected to [{ip_address}:{port}]")
        except Exception as e:
            logging.error(f"Unable to connect to the microscope: {e}")
            self.microscope = None