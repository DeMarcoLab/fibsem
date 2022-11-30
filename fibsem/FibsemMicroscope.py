from abc import ABC, abstractmethod
import fibsem.utils as utils
from pathlib import Path
import os
import logging
from autoscript_sdb_microscope_client import SdbMicroscopeClient

class FibsemMicroscope(ABC):
    """Abstract class containing all the core microscope functionalities"""    
    @abstractmethod
    def connect(self, host: str, port: int):
        pass

    @abstractmethod
    def disconnect(self):
        pass


class ThermoMicroscope(FibsemMicroscope):
    """ThermoFisher Microscope class, uses FibsemMicroscope as blueprint 

    Args:
        FibsemMicroscope (ABC): abstract implementation
    """
    def __init__(self, connection  = None):
        self.connection = connection

    def disconnect(self):
        pass
    
    @classmethod
    def connect(cls, ip_address: str, port: int = 7520):
        """Connect to the FIBSEM microscope."""
        try:
            # TODO: get the port
            logging.info(f"Microscope client connecting to [{ip_address}:{port}]")
            connection = SdbMicroscopeClient()
            connection.connect(host=ip_address, port=port)
            logging.info(f"Microscope client connected to [{ip_address}:{port}]")
        except Exception as e:
            logging.error(f"Unable to connect to the microscope: {e}")
            connection = None
        scope = cls()
        scope.connection = connection
        
        return scope
        # return cls(microscope)