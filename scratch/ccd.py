
from fibsem import utils, acquire
from fibsem.structures import BeamType

import numpy as np
import matplotlib.pyplot as plt



microscope, settings = utils.setup_session(ip_address="10.0.0.1", manufacturer="Thermo")

import napari
viewer = napari.Viewer()


from pprint import pprint 

microscope.connection.imaging.set_active_view(4)
microscope.connection.imaging.set_active_device(3) #CCD = DEVICE 3, BUT VIEW=4 USUALLY WHAT?

import time
from fibsem.structures import FibsemImage


from napari.qt.threading import thread_worker


@thread_worker
def _acquire_image(microscope):
    i = 0
    images = []
    while True:

        print(f"Acquiring image... {i:04d}")
        image = microscope.connection.imaging.get_image()
        image = FibsemImage(image.data, None)
        # images.append(image)

        yield image
        
        time.sleep(0.2)
        i += 1
        if i > 100:
            break


def _on_yield(image):
    try:
        viewer.layers["data"].data = image.data
    except KeyError:
        viewer.add_image(image.data, name="data")


worker = _acquire_image(microscope)
worker.yielded.connect(_on_yield)
worker.finished.connect(lambda: print("Finished!"))
worker.finished.connect(viewer.close)
worker.start()

