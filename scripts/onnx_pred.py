#!/usr/bin/python3
#
#   onnx_pred.py - inference using ONNX model
#   author: Christopher JF Cameron
#

import argparse
import cv2  # type: ignore
import numpy as np  # type: ignore
import os

from skimage.util.shape import view_as_windows  # type: ignore
from onnxruntime import InferenceSession  # type: ignore


def load_onnx_model(model_path: str):
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


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Lamella prediction with ONNX model")
    parser.add_argument("model_path", type=str, help="Path to the ONNX model file")
    parser.add_argument("img_path", type=str, help="Path to the input image file")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="Path to save the output image (default: img directory)",
    )

    return parser.parse_args()


def standardize(img: object, sigma: float = 24.0):
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


def main(args):
    """
    Main function for ONNX prediction.
    #   usage
    #   python onnx_pred.py /path/to/model.onnx /path/to/image.jpeg -o /path/to/output

    Args:
        args (argparse.Namespace): The command line arguments.

    Returns:
        None
    """

    print("Loading ONNX model ... ", end="", flush=True)
    #   load ONNX model
    session, input_name, window_shape, output_name = load_onnx_model(args.model_path)
    stride = window_shape[0] // 5
    print("done")

    print("Loading image ... ", end="", flush=True)
    #   load image
    basename = os.path.basename(args.img_path).replace(".jpeg", "")
    img = cv2.imread(args.img_path).astype(np.float32)
    #   save original image shape
    h, w, c = img.shape
    #   PaddleSeg models expect 3 channel input image
    assert c == 3, f"image must have 3 channels. Found: {c}"
    del c
    print("done")

    #   standardize image
    img = standardize(img)

    #   transpose dimensions from HWC to CHW
    img = np.transpose(img, (2, 0, 1))

    #   pad image for sliding window
    pad_h = max(window_shape[0] - h % stride, 0)
    pad_w = max(window_shape[1] - w % stride, 0)
    img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    _, pad_h, pad_w = img.shape
    del _

    #   window input image
    windows = view_as_windows(
        img, (3, window_shape[0], window_shape[1]), step=stride
    ).squeeze()

    print("Predicting ... ", end="", flush=True)
    #   predict each window
    container = None
    count = np.zeros([1, 1, pad_h, pad_w])
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):

            window = windows[i, j]
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + window_shape[0]
            w_end = w_start + window_shape[1]

            #   add batch dimension to window
            logits = session.run([output_name], {input_name: window[np.newaxis, ...]})[
                0
            ].squeeze()
            del window

            #   add logits to container
            if container is None:
                container = np.zeros([1, logits.shape[0], pad_h, pad_w])
            container[:, :, h_start:h_end, w_start:w_end] += logits
            count[:, :, h_start:h_end, w_start:w_end] += 1
            del h_start, w_start, h_end, w_end, logits
    del img, pad_h, pad_w, i, j
    assert (
        np.min(count) == 1
    ), "There are pixels not predicted. Check window and stride size."

    #   average the predictions
    container = np.argmax(container / count, axis=1).squeeze() * 127
    del count

    #   crop image to remove padding
    container = container[:h, :w]
    del h, w
    print("done")

    import matplotlib.pyplot as plt
    plt.imshow(container, cmap="gray")
    plt.show()

    #   write to storage
    out_path = os.path.join(args.output_dir, f"{basename}.png")
    cv2.imwrite(out_path, container)
    del basename


if __name__ == "__main__":
    args = parse_args()

    #   validate arguments
    assert os.path.exists(args.model_path), "model path does not exist"
    assert os.path.exists(args.img_path), "image path does not exist"
    if args.output_dir is not None:
        print(f"warning - setting output directory to image directory: {args.img_path}")
        args.output_dir = os.path.dirname(args.img_path)

    main(args)
