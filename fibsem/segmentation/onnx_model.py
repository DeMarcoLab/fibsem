

import os
import numpy as np
import onnx
import onnxruntime

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
    
    from fibsem.segmentation.model import load_model
    import torch

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