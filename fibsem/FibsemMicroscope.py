from abc import ABC, abstractmethod

class FibsemMicroscope(ABC):

    @abstractmethod
    def connect(self, host: str = "10.0.0.1", port: int = 7520):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def get_microscope_state(self):
        pass
    
    @abstractmethod
    def patterning(self):
        pass

    @abstractmethod
    def get_needle_position(self):
        pass

    @abstractmethod
    def move_needle(self, position):
        pass

    @abstractmethod
    def insert_needle(self):
        pass

    @abstractmethod
    def retract_needle(self):
        pass    

    @abstractmethod
    def get_stage_position(self):
        pass

    @abstractmethod
    def move_stage(self, position):
        pass        

    @abstractmethod
    def get_eb_settings(self):
        pass

    @abstractmethod
    def set_eb_settings(self, settings):
        pass

    @abstractmethod
    def get_ib_settings(self):
        pass

    @abstractmethod
    def set_ib_settings(self, settings):
        pass    