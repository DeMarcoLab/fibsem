"""Utility functions for pattern analysis and coordinate conversion."""

import logging
import numpy as np
from typing import List, Tuple

from fibsem.milling.base import FibsemMillingStage
from fibsem.structures import (
    FibsemCircleSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemRectangleSettings,
    Point,
    FibsemRectangle,
)

def create_pattern_mask(stage: FibsemMillingStage, image: FibsemImage, include_exclusions: bool = False) -> np.ndarray:
    """Create a binary mask for a single milling stage pattern.
    
    Args:
        stage: FibsemMillingStage to create mask for.
        image: FibsemImage for coordinate conversion and mask dimensions.
        include_exclusions: Whether to include exclusion patterns in the mask.
    
    Returns:
        Binary mask as numpy array with same shape as image.
    """
    image_shape = image.data.shape
    stage_mask = np.zeros(image_shape, dtype=bool)
    
    try:
        shapes = stage.pattern.define()
        
        for shape in shapes:
            # Skip exclusion patterns unless explicitly requested
            if hasattr(shape, 'is_exclusion') and shape.is_exclusion and not include_exclusions:
                continue
            
            # Import here to avoid circular import
            from .plotting import draw_pattern_shape
            drawn_pattern = draw_pattern_shape(shape, image)
            shape_mask = np.zeros(image_shape, dtype=bool)
            
            # Place the pattern in the image mask
            w = drawn_pattern.pattern.shape[1] // 2
            h = drawn_pattern.pattern.shape[0] // 2
            pos = drawn_pattern.position
            
            xmin, xmax = max(0, pos.x - w), min(image_shape[1], pos.x + w)
            ymin, ymax = max(0, pos.y - h), min(image_shape[0], pos.y + h)
            
            if xmax > xmin and ymax > ymin:
                pattern_h = min(2*h, ymax-ymin)
                pattern_w = min(2*w, xmax-xmin)
                shape_mask[ymin:ymin+pattern_h, xmin:xmin+pattern_w] = \
                    drawn_pattern.pattern[:pattern_h, :pattern_w].astype(bool)
            
            stage_mask |= shape_mask
        
    except Exception as e:
        logging.debug(f"Failed to create mask for pattern {stage.pattern.name}: {e}")
    
    return stage_mask


def get_pattern_bounding_box(stage: FibsemMillingStage, image: FibsemImage, expand_percent: float = 0.0) -> Tuple[int, int, int, int]:
    """Get the bounding box of a pattern in image pixel coordinates.
    
    Args:
        stage: FibsemMillingStage to get bounding box for.
        image: FibsemImage for coordinate conversion.
        expand_percent: Percentage to expand bounding box by (e.g., 10.0 for 10%).
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in image pixel coordinates.
        Returns (0, 0, 0, 0) if pattern has no area or fails to process.
    """
    try:
        # Create mask for the pattern
        mask = create_pattern_mask(stage, image, include_exclusions=False)
        
        # Find coordinates where mask is True
        coords = np.where(mask)
        
        if len(coords[0]) == 0:
            # No pattern area found
            return (0, 0, 0, 0)
        
        # Get bounding box coordinates
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Expand bounding box if requested
        if expand_percent > 0:
            width = x_max - x_min
            height = y_max - y_min
            
            # Calculate expansion in pixels
            expand_x = int(width * expand_percent / 100)
            expand_y = int(height * expand_percent / 100)
            
            # Apply expansion while staying within image bounds
            image_height, image_width = image.data.shape
            x_min = max(0, x_min - expand_x)
            y_min = max(0, y_min - expand_y)
            x_max = min(image_width - 1, x_max + expand_x)
            y_max = min(image_height - 1, y_max + expand_y)
        
        return (x_min, y_min, x_max, y_max)
        
    except Exception as e:
        logging.debug(f"Failed to calculate bounding box for pattern {stage.pattern.name}: {e}")
        return (0, 0, 0, 0)


def get_patterns_bounding_box(stages: List[FibsemMillingStage], image: FibsemImage, expand_percent: float = 0.0) -> Tuple[int, int, int, int]:
    """Get the combined bounding box of multiple patterns in image pixel coordinates.
    
    Args:
        stages: List of FibsemMillingStage to get combined bounding box for.
        image: FibsemImage for coordinate conversion.
        expand_percent: Percentage to expand bounding box by (e.g., 10.0 for 10%).
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in image pixel coordinates.
        Returns (0, 0, 0, 0) if no patterns have area.
    """
    if not stages:
        return (0, 0, 0, 0)
    
    # Get bounding boxes for all patterns
    all_boxes = [get_pattern_bounding_box(stage, image, expand_percent) for stage in stages]
    
    # Filter out empty boxes
    valid_boxes = [box for box in all_boxes if box != (0, 0, 0, 0)]
    
    if not valid_boxes:
        return (0, 0, 0, 0)
    
    # Find overall bounding box
    x_mins, y_mins, x_maxs, y_maxs = zip(*valid_boxes)
    
    return (min(x_mins), min(y_mins), max(x_maxs), max(y_maxs))


def bbox_to_normalized_coords(bbox: Tuple[int, int, int, int], image: FibsemImage) -> Tuple[float, float, float, float]:
    """Convert bounding box from pixel coordinates to normalized coordinates (0-1).
    
    Args:
        bbox: Tuple of (x_min, y_min, x_max, y_max) in pixel coordinates.
        image: FibsemImage to get dimensions from.
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in normalized coordinates (0.0-1.0).
        Returns (0.0, 0.0, 0.0, 0.0) if bbox is empty.
    """
    x_min, y_min, x_max, y_max = bbox
    
    # Check for empty bounding box
    if bbox == (0, 0, 0, 0):
        return (0.0, 0.0, 0.0, 0.0)
    
    # Get image dimensions
    image_height, image_width = image.data.shape
    
    # Normalize coordinates (0=0.0, shape=1.0)
    x_min_norm = x_min / image_width
    y_min_norm = y_min / image_height
    x_max_norm = x_max / image_width
    y_max_norm = y_max / image_height
    
    return (x_min_norm, y_min_norm, x_max_norm, y_max_norm)


def normalized_coords_to_bbox(norm_bbox: Tuple[float, float, float, float], image: FibsemImage) -> Tuple[int, int, int, int]:
    """Convert normalized coordinates (0-1) to pixel bounding box coordinates.
    
    Args:
        norm_bbox: Tuple of (x_min, y_min, x_max, y_max) in normalized coordinates (0.0-1.0).
        image: FibsemImage to get dimensions from.
    
    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in pixel coordinates.
        Returns (0, 0, 0, 0) if norm_bbox is empty.
    """
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = norm_bbox
    
    # Check for empty normalized bounding box
    if norm_bbox == (0.0, 0.0, 0.0, 0.0):
        return (0, 0, 0, 0)
    
    # Get image dimensions
    image_height, image_width = image.data.shape
    
    # Convert to pixel coordinates
    x_min = int(x_min_norm * image_width)
    y_min = int(y_min_norm * image_height)
    x_max = int(x_max_norm * image_width)
    y_max = int(y_max_norm * image_height)
    
    # Ensure coordinates are within bounds
    x_min = max(0, min(x_min, image_width - 1))
    y_min = max(0, min(y_min, image_height - 1))
    x_max = max(0, min(x_max, image_width - 1))
    y_max = max(0, min(y_max, image_height - 1))
    
    return (x_min, y_min, x_max, y_max)

def get_pattern_reduced_area(stage: FibsemMillingStage, image: FibsemImage, expand_percent: int = 20) -> FibsemRectangle:
    """Get the bounding box of the pattern in the image, expanded by a percentage.
    Args:
        stage: FibsemMillingStage to get bounding box for.
        image: FibsemImage for coordinate conversion.
        expand_percent: Percentage to expand bounding box by (default 20%).
    Returns:
        FibsemRectangle with normalized coordinates (0.0-1.0)."""
    bbox = get_pattern_bounding_box(stage=stage, image=image, expand_percent=expand_percent)
    xmin, ymin, xmax, ymax = bbox_to_normalized_coords(bbox, image)
    return FibsemRectangle(left=xmin, top=ymin, width=xmax-xmin, height=ymax-ymin)
