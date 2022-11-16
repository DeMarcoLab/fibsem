import fibsem
import numpy as np

class fibsemImage():
    def __init__(
        self,
        data: np.ndarray,
        metadata: dict
    ):
        self.__construct_image_data(data)
        self.metadata = metadata


    def __construct_image_data(data):
        
