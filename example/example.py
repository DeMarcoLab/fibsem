

from fibsem import utils, acquire

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg', force=True) # Activate 'agg' backend for off-screen plotting.


def main():

    # connect to microscope
    microscope, settings = utils.setup_session(manufacturer="Demo", ip_address="localhost")

    # take image with both beams
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)

    # show images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(eb_image.data, cmap="gray")
    ax[0].set_title("Electron Beam Image")
    ax[0].axis("off")
    ax[1].imshow(ib_image.data, cmap="gray")
    ax[1].set_title("Ion Beam Image")
    ax[1].axis("off")
    plt.show()


if __name__ == "__main__":
    main()
