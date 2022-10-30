

from fibsem import utils, acquire
import matplotlib.pyplot as plt


def main():

    # connect to microscope
    microscope, settings = utils.setup_session()

    # take image with both beams
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)

    # show images

    fig, ax = plt.subplots(1, 2, figsize=(7, 5))
    ax[0].imshow(eb_image.data, cmap="gray")
    ax[1].imshow(ib_image.data, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
