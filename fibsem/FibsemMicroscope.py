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
    def __init__(self):
        self.microscope = SdbMicroscopeClient()

    def disconnect(self):
        pass
    
    @classmethod
    def connect(cls, ip_address: str, port: int = 7520):
        """Connect to the FIBSEM microscope."""
        try:
            # TODO: get the port
            logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
            microscope = SdbMicroscopeClient()
            microscope.connect(host=ip_address, port=port)
            logging.info(f"Microscope client connected to [{ip_address}:{port}]")
        except Exception as e:
            logging.error(f"Unable to connect to the microscope: {e}")
            microscope = None
        return cls(microscope)