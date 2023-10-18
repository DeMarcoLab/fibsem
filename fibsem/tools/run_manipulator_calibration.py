from fibsem import utils, calibration, acquire

from fibsem.structures import BeamType, FibsemManipulatorPosition
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    microscope, settings = utils.setup_session()

    # if argv 1 is "debug", run in debug mode
    _DEBUG = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "debug":
            _DEBUG = True

    if _DEBUG:
        microscope.set("scan_rotation", np.deg2rad(0), beam_type=BeamType.ION)
        microscope.move_manipulator_to_position_offset(offset=FibsemManipulatorPosition(), name="EUCENTRIC")
        microscope.move_manipulator_corrected(dx=-50e-6, dy=-150e-6, beam_type=BeamType.ELECTRON)

        settings.image.autocontrast = True
        settings.image.hfw = 900e-6
        eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(eb_image.data, cmap="gray")
        ax[1].imshow(ib_image.data, cmap="gray")

        plt.show()

    calibration._calibrate_manipulator_thermo(microscope, settings, None)

if __name__ == "__main__":
    main()