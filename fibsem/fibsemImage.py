import numpy as np
import tifffile as tff
from dataclasses import dataclass
import json
from fibsem.structures import ImageSettings, BeamType, GammaSettings, Point, MicroscopeState, BeamSettings, FibsemRectangle
from autoscript_sdb_microscope_client.structures import (AdornedImage, StagePosition, ManipulatorPosition)
from fibsem.config import METADATA_VERSION

THERMO_ENABLED = True
if THERMO_ENABLED:
    from autoscript_sdb_microscope_client.structures import AdornedImage


@dataclass
class FibsemImageMetadata:
    """Metadata for a FibsemImage."""

    image_settings: ImageSettings
    pixel_size: Point
    microscope_state: MicroscopeState
    version: str = METADATA_VERSION

    def __to_dict__(self) -> dict:
        """Converts metadata to a dictionary.

        Returns:
            dictionary: self as a dictionary
        """
        settings_dict = self.image_settings.__to_dict__()
        settings_dict["version"] = self.version
        settings_dict["pixel_size"] = self.pixel_size.__to_dict__()
        settings_dict["microscope_state"] = self.microscope_state.__to_dict__()
        return settings_dict

    @staticmethod
    def __from_dict__(settings: dict) -> "ImageSettings":
        """Converts a dictionary to metadata."""

        image_settings = ImageSettings(
            resolution=settings["resolution"],
            dwell_time=settings["dwell_time"],
            hfw=settings["hfw"],
            autocontrast=settings["autocontrast"],
            beam_type=BeamType[settings["beam_type"].upper()],
            gamma=GammaSettings.__from_dict__(settings["gamma"]),
            save=settings["save"],
            save_path=settings["save_path"],
            label=settings["label"],
            reduced_area=FibsemRectangle.__from_dict__(settings["reduced_area"]),
        )
        version = settings["version"]
        pixel_size = Point.__from_dict__(settings["pixel_size"])
        microscope_state = MicroscopeState(
            timestamp=settings["microscope_state"]["timestamp"],
            absolute_position=StagePosition(),
            eb_settings=BeamSettings.__from_dict__(settings["microscope_state"]["eb_settings"]),
            ib_settings=BeamSettings.__from_dict__(settings["microscope_state"]["ib_settings"]),
        )

        metadata = FibsemImageMetadata(
            image_settings=image_settings,
            version=version,
            pixel_size=pixel_size,
            microscope_state=microscope_state,
        )
        return metadata


class FibsemImage:
    def __init__(self, data: np.ndarray, metadata: FibsemImageMetadata = None):
        self.data = check_data_format(data)
        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = None

    @classmethod
    def load(cls, tiff_path: str) -> "FibsemImage":
        """Loads a FibsemImage from a tiff file.

        Args:
            tiff_path (path): path to the tif* file

        Returns:
            FibsemImage: instance of FibsemImage
        """
        with tff.TiffFile(tiff_path) as tiff_image:
            data = tiff_image.asarray()
            try:
                metadata = json.loads(
                    tiff_image.pages[0].tags["ImageDescription"].value
                )
                metadata = FibsemImageMetadata.__from_dict__(metadata)
            except:
                metadata = None
        return cls(data=data, metadata=metadata)

    def save(self, save_path: str) -> None:
        """Saves a FibsemImage to a tiff file.

        Inputs:
            save_path (path): path to save directory and filename
        """
        if self.metadata is not None:
            metadata_dict = self.metadata.__to_dict__()
        else:
            metadata_dict = None
        tff.imwrite(
            save_path,
            self.data,
            metadata=metadata_dict,
        )

    @classmethod
    def fromAdornedImage(
        cls, adorned: AdornedImage, image_settings: ImageSettings
    ) -> "FibsemImage":
        """Creates FibsemImage from an AdornedImage (microscope output format).

        Args:
            adorned (AdornedImage): Adorned Image from microscope
            metadata (FibsemImageMetadata, optional): metadata extracted from microscope output. Defaults to None.

        Returns:
            FibsemImage: instance of FibsemImage from AdornedImage
        """

        microscope = MicroscopeState(
            timestamp=adorned.metadata.acquisition.acquisition_datetime,
            absolute_position=StagePosition(),
            eb_settings=BeamSettings(beam_type=BeamType.ELECTRON),
            ib_settings=BeamSettings(beam_type=BeamType.ION),
        )
        pixel_size = Point(adorned.metadata.binary_result.pixel_size.x, adorned.metadata.binary_result.pixel_size.y)
        metadata=FibsemImageMetadata(image_settings=image_settings, pixel_size=pixel_size, microscope_state=microscope)
        return cls(data=adorned.data, metadata=metadata)


def check_data_format(data: np.ndarray) -> np.ndarray:
    """Checks that data is in the correct format."""
    assert data.ndim == 2  # or data.ndim == 3
    assert data.dtype == np.uint8
    # if data.ndim == 3 and data.shape[2] == 1:
    #     data = data[:, :, 0]
    return data
