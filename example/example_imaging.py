import matplotlib
import matplotlib.pyplot as plt

from fibsem import acquire, utils
from fibsem.structures import BeamType
import logging

matplotlib.use('TkAgg', force=True) # Activate 'agg' backend for off-screen plotting.


"""
This script will take an image with the electron beam, an image with the ion beam, and an image with both beams. 
The images are then displayed in a matplotlib figure.

The settings for images are stored in the settings.image struct, and can be modified before taking an image.

For more detail on the settings, see the documentation for the ImageSettings class.

"""

def main():
    
    # connect to the microscope
    microscope, settings = utils.setup_session(manufacturer="Demo", ip_address="localhost")

    # info about ImageSettings
    logging.info(f"\nAcquiring Images Example:")
    logging.info(f"The current image settings are: \n{settings.image}")

    # take an image with the electron beam
    settings.image.beam_type = BeamType.ELECTRON
    eb_image = acquire.new_image(microscope, settings.image)

    # take an image with the ion beam
    settings.image.beam_type = BeamType.ION
    ib_image = acquire.new_image(microscope, settings.image)

    # take an image with both beams with increased hfw
    settings.image.hfw = 400e-6       
    ref_eb_image, ref_ib_image = acquire.take_reference_images(microscope, settings.image)

    # show images

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0][0].imshow(eb_image.data, cmap="gray")
    ax[0][0].set_title("Electron Image 01")
    ax[0][1].imshow(ib_image.data, cmap="gray")
    ax[0][1].set_title("Ion Image 01")
    ax[1][0].imshow(ref_eb_image.data, cmap="gray")
    ax[1][0].set_title("Electron Image 02 (Reference)")
    ax[1][1].imshow(ref_ib_image.data, cmap="gray")
    ax[1][1].set_title("Ion Image 02 (Reference)")
    plt.show()


if __name__ == "__main__":
    main()
