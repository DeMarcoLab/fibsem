import numpy as np
import torch
from transformers import SamModel, SamProcessor


class SamModelWrapper:
    def __init__(self, checkpoint: str = "facebook/sam-vit-base", device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model = SamModel.from_pretrained(checkpoint).to(self.device)
        processor = SamProcessor.from_pretrained(checkpoint)

        self.checkpoint = checkpoint
        self.model = model
        self.processor = processor

    def __call__(
        self,
        image: np.ndarray,
        points: list[list[list[int]]] = None,
        labels: list[list[bool]] = None,
        input_masks=None,
        multimask_output: bool = False,
    ):
        inputs = self.processor(
            image,
            input_points=points,
            input_labels=labels,
            return_tensors="pt",
            input_masks=input_masks,
            multimask_output=multimask_output,
        ).to(self.device)
        # TODO: multi-mask output doesn't seem to work?
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs, inputs

    # to mimic the functionality of sam predictor
    def predict(
        self,
        image,
        points: list[list[list[int]]] = None,
        labels: list[list[bool]] = None,
        input_masks: np.ndarray = None,
        multimask_output: bool = False,
    ):
        outputs, inputs = self(image, points, labels, input_masks, multimask_output)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        scores = outputs.iou_scores
        logits = outputs.pred_masks

        # get best mask
        idx = np.argmax(scores.detach().cpu().numpy())
        mask = masks[0][0][idx].detach().cpu().numpy()
        score = scores[0][0][idx].detach().cpu().numpy()
        logits = logits[0][0][idx].detach().cpu().unsqueeze(0).numpy()

        return mask, score, logits

    def __repr__(self):
        return f"SamModelWrapper({self.checkpoint}, {self.device})"
