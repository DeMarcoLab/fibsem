from fibsem import utils, acquire
from fibsem.structures import BeamType, FibsemStagePosition, FibsemPatternSettings, FibsemPattern
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg', force=True) # Activate 'agg' backend for off-screen plotting.


def connect_to_microscope():

    #### connect to microscope ######
    microscope, settings = utils.setup_session()
    

    return microscope,settings


def imaging_1(microscope,settings):

    image_settings = settings.image 

    ##### take image with both beams #####
    ref_eb_image, ref_ib_image = acquire.take_reference_images(microscope, image_settings)

    ##### take image with only electron beam #####
    image_settings.beam_type = BeamType.ELECTRON
    eb_image = acquire.new_image(microscope, image_settings)

    ##### take image with only ion beam #####
    image_settings.beam_type = BeamType.ION
    ib_image = acquire.new_image(microscope, image_settings)

    # show images

    fig, ax = plt.subplots(2, 2, figsize=(7, 5))
    ax[0][0].set_title("Reference Electron Beam Image")
    ax[1][0].set_title("Reference Ion Beam Image")
    ax[0][0].imshow(ref_eb_image.data, cmap="gray")
    ax[1][0].imshow(ref_ib_image.data, cmap="gray")
    ax[0][1].set_title("Individual Electron Beam Image")
    ax[1][1].set_title("Individual Ion Beam Image")
    ax[0][1].imshow(eb_image.data, cmap="gray")
    ax[1][1].imshow(ib_image.data, cmap="gray")
    plt.show()


def imaging_2(microscope,settings):

    image_settings = settings.image

    ## in this example, the HFW will be changed 

    ref_eb, ref_ib = acquire.take_reference_images(microscope=microscope,image_settings=image_settings)

    old_hfw = image_settings.hfw

    #double the old HFW

    new_hfw = 2*old_hfw

    image_settings.hfw = new_hfw

    new_eb, new_ib = acquire.take_reference_images(microscope=microscope,image_settings=image_settings)

    fig, ax = plt.subplots(2, 2, figsize=(7, 5))
    ax[0][0].set_title("Reference Electron Beam Image")
    ax[1][0].set_title("Reference Ion Beam Image")
    ax[0][0].imshow(ref_eb.data, cmap="gray")
    ax[1][0].imshow(ref_ib.data, cmap="gray")
    ax[0][1].set_title("Electron Beam Image New HFW")
    ax[1][1].set_title("Ion Beam Image New HFW")
    ax[0][1].imshow(new_eb.data, cmap="gray")
    ax[1][1].imshow(new_ib.data, cmap="gray")
    plt.show()

## Parameters that can be changed in image settings
#   -Beamtype
#   -Autocontrast
#   -Horizontal Field Width (HFW)
#   -Gamma 
#   -Dwell Time
#   -Resolution
#   -Save Path
#   -Autosave


if __name__ == "__main__":

    microscope,settings = connect_to_microscope()

    #imaging_1(microscope=microscope,settings=settings)

    imaging_2(microscope=microscope,settings=settings)
