

import os
import logging

import cv2
import numpy as np
import onnx
import onnxruntime
import PIL.Image
from onnxruntime import InferenceSession
from skimage.util.shape import view_as_windows
import itertools
from tqdm import tqdm
import concurrent.futures

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
    num_output_classes = session.get_outputs()[0].shape[1]

    return session, input_name, window_shape, output_name, num_output_classes

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
    """
    ONNX model for windowed inference.
    This code refactor enables multi-threaded inference and Gaussian weighted windows
    Improves model inference, inference time and window edge effects"""
    def __init__(self, checkpoint: str = None):
        if checkpoint is not None:
            self.load_model(checkpoint)
        self.device = None
        self.num_workers = min(8, os.cpu_count())
    
    def load_model(self, checkpoint="autolamella-mega.onnx"):
        # download checkpoint if needed
        # checkpoint = download_checkpoint(checkpoint)
        self.checkpoint = os.path.basename(checkpoint)

        # load inference session
        session = load_windowed_onnx_model(checkpoint)
        self.session, self.input_name, self.window_shape, self.output_name, self.num_output_classes = session
    
    def GaussianWeightMatrix(self, window_shape: tuple[int, int]) -> np.ndarray:
        """
        Generate a Gaussian weight matrix for the sliding window."""
        ksize = min(window_shape)
        w_matrix = cv2.getGaussianKernel(
            ksize=ksize, sigma=ksize / 4
        )  #  sigma may need to be adjust based on window size
        w_matrix = np.outer(w_matrix, w_matrix)
        #   normalize kernel
        w_matrix = w_matrix / np.max(w_matrix)
        w_matrix = w_matrix[np.newaxis, ...]
        del ksize
        return w_matrix
    
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
        windows = view_as_windows(
        img, (3, self.window_shape[0], self.window_shape[1]), step=stride
        ).reshape(-1, 3, self.window_shape[0], self.window_shape[1])

        logging.debug(f"pre_process: {img.shape}, {windows.shape}, {h}, {w}, {pad_h}, {pad_w}, {stride}")

        return img, windows, h, w, pad_h, pad_w, stride
    
    def window_indices(self,pad_h: int, pad_w: int, window_shape: tuple[int, int], stride: int, windows: np.ndarray) -> None:
        #   determine the i and j indices for each window
        num_windows_h = ((pad_h - window_shape[0]) / stride) + 1
        num_windows_w = ((pad_w - window_shape[1]) / stride) + 1
        indices_i = [i * stride for i in range(int(num_windows_h))]
        indices_j = [j * stride for j in range(int(num_windows_w))]
        indices = list(itertools.product(indices_i, indices_j))
        assert len(indices) == windows.shape[0], "Indices do not match windows shape."
        del num_windows_h, num_windows_w, indices_i
        return indices

    def worker_process(
    self,session: InferenceSession, window: np.ndarray, index: tuple[int, int]
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """
        Worker process for multi-threaded ONNX inference.

        Args:
            session (InferenceSession): The ONNX model session.
            window (np.ndarray): The input image window.
            index (tuple[int, int]): The i and j indices for the window.

        Returns:
            np.ndarray: The logits for the input window.
            tuple[int, int]: The i and j indices for the window.
        """
        logits = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: window[np.newaxis, ...]},
        )[0].squeeze()

        return logits, index
    
    def inference(self, img: np.ndarray, rgb: bool = True) -> np.ndarray:
        """Perform inference on the provided image.
        Args:
            img (np.ndarray): The image to segment.
            rgb (bool): Whether to return an RGB image.
        Returns:
            np.ndarray: The segmented image."""
        pass
        
        w_matrix = self.GaussianWeightMatrix(self.window_shape)

        img, windows, h, w, pad_h, pad_w, stride = self.pre_process(img)

        indices = self.window_indices(pad_h, pad_w, self.window_shape, stride, windows)

        container = np.zeros([1, self.num_output_classes, pad_h, pad_w])
        count = np.zeros([1, 1, pad_h, pad_w])
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            jobs = {
                executor.submit(self.worker_process, self.session, window, idx): (window, idx)
                for (window, idx) in zip(windows, indices)
            }
            for job in concurrent.futures.as_completed(jobs):
                logits, idx = job.result()
                container[
                    :,
                    :,
                    idx[0] : idx[0] + logits.shape[1],
                    idx[1] : idx[1] + logits.shape[2],
                ] += (
                    logits * w_matrix
                )
                count[
                    :,
                    :,
                    idx[0] : idx[0] + logits.shape[1],
                    idx[1] : idx[1] + logits.shape[2],
                ] = w_matrix
        del w_matrix, executor, jobs, logits, idx

        
        mask = np.argmax(container / count, axis=1).squeeze()  # 2d class map
        del container, count
        # crop image to remove padding
        mask = mask[:h, :w]

        if rgb:
            mask = decode_segmap_v2(mask)
        return mask

class SegmentationModelWindowONNX_OLD:
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
        self.session, self.input_name, self.window_shape, self.output_name, self.num_output_classes = session

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
