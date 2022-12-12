

from fibsem import utils, acquire
from fibsem.structures import BeamType, ImageSettings, GammaSettings

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg', force=True) # Activate 'agg' backend for off-screen plotting.


def main():

    # connect to microscope
    microscope, settings = utils.setup_session()

    #settings
    # set gamma settings
    gamma_settings = GammaSettings(
            enabled=True,
            min_gamma=0.5,
            max_gamma=1.8,
            scale_factor=0.01,
            threshold=46,
        )

    # set imaging settings
    image_settings = ImageSettings(
            resolution="1536x1024",
            dwell_time=1.0e-6,
            hfw=900.0e-6,
            autocontrast=True,
            beam_type=BeamType.ION,
            gamma=gamma_settings,
            save=True,
            save_path="fibsem\\test_images",
            label=utils.current_timestamp(),
            reduced_area=None,
        )

    # take image with both beams
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)

    # show images

    fig, ax = plt.subplots(1, 2, figsize=(7, 5))
    ax[0].set_title("Electron Beam Image")
    ax[1].set_title("Ion Beam Image")
    ax[0].imshow(eb_image.data, cmap="gray")
    ax[1].imshow(ib_image.data, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
