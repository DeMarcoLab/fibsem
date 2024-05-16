import numpy as np
import logging
import skimage
import tifffile
#from types import SimpleNamespace
from datetime import datetime

from argparse import Namespace

class SpoofAdornedImageMetadata:

    def __init__(self):
        self._PixelSizeX_m = 1.8e-6
        self._PixelSizeY_m = 1.8e-6

        # binary_result_dict = { "pixel_size": {"x": self._PixelSizeX_m, "y": self._PixelSizeY_m} }
        # self.binary_result = Namespace(**binary_result_dict)
        self.binary_result = Namespace( pixel_size = Namespace(x=self._PixelSizeX_m, y=self._PixelSizeY_m))
        
        # stage_settings_dict = { "stage_position": {
        #     "x":0 , "y":0, "z":0,
        #     "r":0, "t":0
        # }}
        # self.stage_settings = Namespace(**stage_settings_dict)
        self.stage_settings = Namespace( stage_position = Namespace (
            x=0 , y=0, z=0, r=0, t=0
        ))

        # acquisition.acquisition_datetime
        # acquisition_dict = { "acquisition_datetime": datetime.now()}
        # self.acquisition = Namespace(**acquisition_dict)
        self.acquisition = Namespace ( acquisition_datetime = str(datetime.now()) )

    @property
    def PixelSizeX_m(self):
        return self._PixelSizeX_m
    
    @PixelSizeX_m.setter
    def PixelSizeX_m(self,value):
        self._PixelSizeX_m = value
        self.binary_result.pixel_size.x=self._PixelSizeX_m

    @property
    def PixelSizeY_m(self):
        return self._PixelSizeY_m
    
    @PixelSizeX_m.setter
    def PixelSizeY_m(self,value):
        self._PixelSizeX_m = value
        self.binary_result.pixel_size.y=self._PixelSizeY_m

    @property
    def metadata_as_xml(self):
        ret=f"""<?xml version="1.0"?>
<Metadata xmlns:nil="http://schemas.fei.com/Metadata/v1/2013/07" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <BinaryResult>
        <PixelSize>
            <X unit="m" unitPrefixPower="1">{self.PixelSizeX_m}</X>
            <Y unit="m" unitPrefixPower="1">{self.PixelSizeY_m}</Y>
        </PixelSize>
    </BinaryResult>
    <Acquisition>
        <AcquisitionDatetime>2024-03-05T09:58:37</AcquisitionDatetime>
        <BeamType>Electron</BeamType>
        <ColumnType>Elstar</ColumnType>
        <SourceType>FEG</SourceType>
    </Acquisition>
</Metadata>
        """

        return ret

    @property
    def metadata_as_ini(self):
        ret=f"""
[Scan]
PixelWidth={self.PixelSizeX_m}
PixelHeight={self.PixelSizeY_m}

        """

        return ret

# To be used with Demo2Microscope
class SpoofAdornedImage:
    """
    Spoof AdornedImage object

    should contain the following Items:
    raw_data, width, height, raw_encoding, bit_depth, metadata, data, encoding, checksum, thumbnail
    """

    def __init__(self, data2d, tiff_metadata=None, is_thumbnail=False):
        self.height = data2d.shape[0]
        self.width = data2d.shape[1]

        self.data = data2d.copy()

        self.bit_depth = 32
        if isinstance(data2d, np.uint8) or isinstance(data2d, np.int8):
            self.bit_depth = 8
        elif isinstance(data2d, np.uint16) or isinstance(data2d, np.int16):
            self.bit_depth = 16
        # TODO: check bit_depth properly, not defaulting to 32 bit

        self.raw_data = data2d.copy()
        
        sp_ad_im_met = SpoofAdornedImageMetadata()


        if not tiff_metadata is None:
            sp_ad_im_met.PixelSizeX_m = tiff_metadata["Scan"]["PixelWidth"]
            sp_ad_im_met.PixelSizeY_m = tiff_metadata["Scan"]["PixelHeight"]
            logging.info(f"SpoofAdornedImage __init__ with tiff_metadata  PixelWidth_m:{sp_ad_im_met.PixelSizeX_m}, PixelHeight_m:{sp_ad_im_met.PixelSizeY_m}")

        self.metadata = sp_ad_im_met

        self.is_thumbnail = is_thumbnail
        self.thumbnail = None
        if not is_thumbnail:
            # generate thumbnail 256x256
            import skimage.transform

            th_data = skimage.transform.resize_local_mean(self.data, (256, 256))

            self.thumbnail = SpoofAdornedImage(
                th_data, tiff_metadata=tiff_metadata , is_thumbnail=True
            )

        self.checksum = np.sum(self.data)
        self.encoding = 0  # UNSIGNED ImageDataEncoding
        self.raw_encoding = 0

    def save(self, filepath):
        logging.info(f"SpoofAdornedImage: Saving image to {filepath}")
        tifffile.imwrite(filepath, self.data)

    @staticmethod
    def load(filepath):
        # Loads from a TIFF file and returns a new instance of SpoofAdornedImage
        logging.info(f"Loading {filepath} as spoof image")
        with tifffile.TiffFile(filepath) as f:
            metadata0 = f.fei_metadata
            data0 = f.asarray()
            new_ad_im = SpoofAdornedImage(data0, tiff_metadata=metadata0, is_thumbnail=False)

        return new_ad_im
    
