

import os
import logging

import cv2
import numpy as np
import onnx
import onnxruntime
import PIL.Image
from onnxruntime import InferenceSession
from skimage.util.shape import view_as_windows

from fibsem.segmentation.utils import decode_segmap_v2, download_checkpoint

### ONNX


class SegmentationModelONNX:

    def __init__(self, checkpoint: str = None):
        if checkpoint is not None:
            self.load_model(checkpoint)

    def load_model(self, checkpoint="autolamella-mega.onnx"):

        # download checkpoint if needed
        checkpoint = download_checkpoint(checkpoint)
        self.checkpoint = os.path.basename(checkpoint)

        # load inference session
        self.session = onnxruntime.InferenceSession(checkpoint, providers=["CPUExecutionProvider"])


    def inference(self, img: np.ndarray, rgb: bool = True):
        # preprocess
        imgt = img.astype(np.float32)
        imgt = np.expand_dims(imgt, axis=0)
        imgt = np.expand_dims(imgt, axis=0)
        imgt /= 255.0

        # inference
        ort_inputs = {self.session.get_inputs()[0].name: imgt}
        ort_outs = self.session.run(None, ort_inputs)


        # softmax
        outputs = ort_outs[0] # TODO: support batch size > 1
        outputs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
        masks = np.argmax(outputs, axis=1)
        mask = masks[0, :, :]

        # convert to rgb
        if rgb: 
            mask = decode_segmap_v2(mask)
        return mask

def export_model_to_onnx(checkpoint: str, onnx_path: str):
    
    import torch

    from fibsem.segmentation.model import load_model

    # get fibsem model
    model = load_model(checkpoint)
    model.model.to("cpu")
    model.model.eval()

    # From https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    batch_size = 1
    x = torch.randn(batch_size, 1, 1024, 1536, requires_grad=True, dtype=torch.float)
    torch_out = model.model(x)

    # Export the model
    torch.onnx.export(model.model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    onnx_path,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})


    # load onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    # load inference session
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

# export_model_to_onnx("autolamella-mega-latest.pt", "autolamella-mega-20231230.onnx")

## PPLITESEG WINDOWED MODEL
def load_windowed_onnx_model(model_path: str) -> tuple:
    """
    Load the ONNX model.

    Args:
        model_path (str): File path to the ONNX model.

    Returns:
        InferenceSession: The ONNX model session.
        str: The name of the input tensor.
        Tuple[int, int]: The shape of the input tensor.
        str: The name of the output tensor.
    """
    session = InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    window_shape = session.get_inputs()[0].shape[2:]
    output_name = session.get_outputs()[0].name

    return session, input_name, window_shape, output_name

def standardize(img: object, sigma: float = 24.0) -> np.ndarray:
    """
    Standardize the pixel intensities of the provided image.

    Args:
        img (np.ndarray): The image to standardize.
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        np.ndarray: The standardized image.
    """
    #   subtract local mean
    smooth = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    img = np.subtract(img, smooth)
    # scale pixel intensities
    img = img / np.std(img)
    del smooth

    return img.astype(np.float32)


class SegmentationModelWindowONNX:
    def __init__(self, checkpoint: str = None):
        if checkpoint is not None:
            self.load_model(checkpoint)
        self.device = None

    def load_model(self, checkpoint="autolamella-mega.onnx"):
        # download checkpoint if needed
        # checkpoint = download_checkpoint(checkpoint)
        self.checkpoint = os.path.basename(checkpoint)

        # load inference session
        session = load_windowed_onnx_model(checkpoint)
        self.session, self.input_name, self.window_shape, self.output_name = session

    def pre_process(self, img: np.ndarray) -> np.ndarray:
        """Pre-process the image for inference, calculate window parameters"""
        ##### PREPROCESSING
        if img.ndim == 2:  # 2d grayscale image -> 3d rgb (grayscale)
            img = np.array(PIL.Image.fromarray(img).convert("RGB"))

        if img.dtype != np.float32:
            img = img.astype(np.float32)

        # image and window parameters
        h, w, c = img.shape
        stride = self.window_shape[0] // 5  # MAGIC_NUMBER

        #   standardize image
        img = standardize(img)

        #   transpose dimensions from HWC to CHW
        img = np.transpose(img, (2, 0, 1))

        #   pad image for sliding window
        pad_h = max(self.window_shape[0] - h % stride, 0)
        pad_w = max(self.window_shape[1] - w % stride, 0)
        img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
        _, pad_h, pad_w = img.shape

        #   window input image
        windows: np.ndarray = view_as_windows(
            img, (3, self.window_shape[0], self.window_shape[1]), step=stride
        ).squeeze()

        logging.debug(f"pre_process: {img.shape}, {windows.shape}, {h}, {w}, {pad_h}, {pad_w}, {stride}")

        return img, windows, h, w, pad_h, pad_w, stride

    def inference(self, img: np.ndarray, rgb: bool = True) -> np.ndarray:
        """Perform inference on the provided image.
        Args:
            img (np.ndarray): The image to segment.
            rgb (bool): Whether to return an RGB image.
        Returns:
            np.ndarray: The segmented image."""

        # pre-process image
        img, windows, h, w, pad_h, pad_w, stride = self.pre_process(img)

        # inference on each window
        container = None
        count = np.zeros([1, 1, pad_h, pad_w])
        for i in range(windows.shape[0]):
            for j in range(windows.shape[1]):
                window = windows[i, j]
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + self.window_shape[0]
                w_end = w_start + self.window_shape[1]

                # add batch dimension to window
                logits = self.session.run(
                    [self.output_name], 
                    {self.input_name: window[np.newaxis, ...]}
                )[0].squeeze()
                del window

                # add logits to container
                if container is None:
                    container = np.zeros([1, logits.shape[0], pad_h, pad_w])
                container[:, :, h_start:h_end, w_start:w_end] += logits
                count[:, :, h_start:h_end, w_start:w_end] += 1
                del h_start, w_start, h_end, w_end, logits

        assert (
            np.min(count) == 1
        ), "There are pixels not predicted. Check window and stride size."

        # post-process
        # average the predictions across windows
        mask = np.argmax(container / count, axis=1).squeeze()  # 2d class map
        del container, count
        # crop image to remove padding
        mask = mask[:h, :w]

        if rgb:
            mask = decode_segmap_v2(mask)
        return mask
