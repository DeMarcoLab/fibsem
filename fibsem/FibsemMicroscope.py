from abc import ABC, abstractmethod

class FibsemMicroscope(ABC):

    @abstractmethod
    def connect(self, host: str = "10.0.0.1", port: int = 7520):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    