"""Tests for pattern utility functions."""

import numpy as np
import pytest
from fibsem.structures import (
    FibsemImage, 
    FibsemRectangleSettings, 
    FibsemCircleSettings,
    Point
)
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern, CirclePattern
from fibsem.milling.patterning.utils import (
    create_pattern_mask,
    get_pattern_bounding_box,
    get_patterns_bounding_box,
    bbox_to_normalized_coords,
    normalized_coords_to_bbox,
)


class TestBoundingBoxFunctions:
    """Test bounding box utility functions."""
    
    def test_get_pattern_bounding_box_rectangle(self):
        """Test bounding box calculation for rectangle pattern."""
        image = FibsemImage.generate_blank_image(resolution=[512, 512], pixel_size=Point(x=1e-9, y=1e-9))
        
        # Create rectangle pattern (20um x 10um) at center
        pattern = RectanglePattern(
            width=20e-6, height=10e-6,
            point=Point(x=0.0, y=0.0)
        )
        stage = FibsemMillingStage(name="Test Rectangle", pattern=pattern)
        
        bbox = get_pattern_bounding_box(stage, image)
        
        # Should return valid coordinates
        assert len(bbox) == 4
        x_min, y_min, x_max, y_max = bbox
        assert x_min < x_max
        assert y_min < y_max
        assert all(coord >= 0 for coord in bbox)
    
    def test_get_pattern_bounding_box_with_expansion(self):
        """Test bounding box expansion functionality."""
        image = FibsemImage.generate_blank_image(resolution=[256, 256], pixel_size=Point(x=1e-9, y=1e-9))
        
        pattern = RectanglePattern(width=10e-6, height=10e-6)
        stage = FibsemMillingStage(name="Test", pattern=pattern)
        
        # Get bbox without expansion
        bbox_normal = get_pattern_bounding_box(stage, image)
        
        # Get bbox with 20% expansion
        bbox_expanded = get_pattern_bounding_box(stage, image, expand_percent=20.0)
        
        # Expanded bbox should be larger
        if bbox_normal != (0, 0, 0, 0):  # Only test if pattern exists
            assert bbox_expanded[0] <= bbox_normal[0]  # x_min smaller or equal
            assert bbox_expanded[1] <= bbox_normal[1]  # y_min smaller or equal
            assert bbox_expanded[2] >= bbox_normal[2]  # x_max larger or equal
            assert bbox_expanded[3] >= bbox_normal[3]  # y_max larger or equal
    
    def test_get_patterns_bounding_box_multiple_patterns(self):
        """Test combined bounding box for multiple patterns."""
        image = FibsemImage.generate_blank_image(resolution=[512, 512], pixel_size=Point(x=1e-9, y=1e-9))
        
        # Create two patterns at different locations
        pattern1 = RectanglePattern(
            width=10e-6, height=10e-6,
            point=Point(x=-50e-6, y=-50e-6)
        )
        pattern2 = CirclePattern(
            radius=5e-6,
            point=Point(x=50e-6, y=50e-6)
        )
        
        stages = [
            FibsemMillingStage(name="Rect", pattern=pattern1),
            FibsemMillingStage(name="Circle", pattern=pattern2)
        ]
        
        combined_bbox = get_patterns_bounding_box(stages, image)
        
        # Should encompass both patterns
        assert len(combined_bbox) == 4
        if combined_bbox != (0, 0, 0, 0):
            x_min, y_min, x_max, y_max = combined_bbox
            assert x_min < x_max
            assert y_min < y_max
    
    def test_get_patterns_bounding_box_empty_list(self):
        """Test combined bounding box with empty stage list."""
        image = FibsemImage.generate_blank_image(resolution=[100, 100], pixel_size=Point(x=1e-9, y=1e-9))
        stages = []
        
        bbox = get_patterns_bounding_box(stages, image)
        assert bbox == (0, 0, 0, 0)


class TestCoordinateConversion:
    """Test coordinate conversion functions."""
    
    def test_bbox_to_normalized_coords_empty(self):
        """Test normalized coordinate conversion for empty bbox."""
        image = FibsemImage.generate_blank_image(resolution=[100, 100], pixel_size=Point(x=1e-9, y=1e-9))
        bbox = (0, 0, 0, 0)
        
        norm_bbox = bbox_to_normalized_coords(bbox, image)
        assert norm_bbox == (0.0, 0.0, 0.0, 0.0)
    
    def test_bbox_to_normalized_coords_full_image(self):
        """Test normalized coordinate conversion for full image."""
        image = FibsemImage.generate_blank_image(resolution=[100, 50], pixel_size=Point(x=1e-9, y=1e-9))  # 100x50 image
        bbox = (0, 0, 99, 49)  # Full image bbox
        
        norm_bbox = bbox_to_normalized_coords(bbox, image)
        expected = (0.0, 0.0, 0.99, 0.98)  # 99/100, 49/50
        assert norm_bbox == expected
    
    def test_bbox_to_normalized_coords_center_quarter(self):
        """Test normalized coordinate conversion for center quarter."""
        image = FibsemImage.generate_blank_image(resolution=[100, 100], pixel_size=Point(x=1e-9, y=1e-9))
        bbox = (25, 25, 75, 75)  # Center quarter
        
        norm_bbox = bbox_to_normalized_coords(bbox, image)
        expected = (0.25, 0.25, 0.75, 0.75)
        assert norm_bbox == expected
    
    def test_normalized_coords_to_bbox_empty(self):
        """Test pixel coordinate conversion for empty normalized bbox."""
        image = FibsemImage.generate_blank_image(resolution=[100, 100], pixel_size=Point(x=1e-9, y=1e-9))
        norm_bbox = (0.0, 0.0, 0.0, 0.0)
        
        bbox = normalized_coords_to_bbox(norm_bbox, image)
        assert bbox == (0, 0, 0, 0)
    
    def test_normalized_coords_to_bbox_center_quarter(self):
        """Test pixel coordinate conversion for center quarter."""
        image = FibsemImage.generate_blank_image(resolution=[100, 100], pixel_size=Point(x=1e-9, y=1e-9))
        norm_bbox = (0.25, 0.25, 0.75, 0.75)
        
        bbox = normalized_coords_to_bbox(norm_bbox, image)
        expected = (25, 25, 75, 75)
        assert bbox == expected
    
    def test_normalized_coords_to_bbox_bounds_checking(self):
        """Test pixel coordinate conversion bounds checking."""
        image = FibsemImage.generate_blank_image(resolution=[100, 100], pixel_size=Point(x=1e-9, y=1e-9))
        # Coordinates that would go out of bounds
        norm_bbox = (1.5, 1.5, 2.0, 2.0)
        
        bbox = normalized_coords_to_bbox(norm_bbox, image)
        # Should be clamped to image bounds
        assert bbox[0] >= 0 and bbox[0] < 100
        assert bbox[1] >= 0 and bbox[1] < 100
        assert bbox[2] >= 0 and bbox[2] < 100
        assert bbox[3] >= 0 and bbox[3] < 100
    
    def test_coordinate_conversion_roundtrip(self):
        """Test that converting bbox->normalized->bbox preserves values."""
        image = FibsemImage.generate_blank_image(resolution=[200, 150], pixel_size=Point(x=1e-9, y=1e-9))
        original_bbox = (20, 30, 180, 120)
        
        # Convert to normalized and back
        norm_bbox = bbox_to_normalized_coords(original_bbox, image)
        final_bbox = normalized_coords_to_bbox(norm_bbox, image)
        
        assert final_bbox == original_bbox
    
    def test_coordinate_conversion_different_image_sizes(self):
        """Test coordinate conversion with different image sizes."""
        # Test that normalized coordinates work across different image sizes
        bbox1 = (10, 10, 90, 90)
        image1 = FibsemImage.generate_blank_image(resolution=[100, 100], pixel_size=Point(x=1e-9, y=1e-9))
        image2 = FibsemImage.generate_blank_image(resolution=[200, 200], pixel_size=Point(x=1e-9, y=1e-9))
        
        # Convert bbox from image1 to normalized
        norm_bbox = bbox_to_normalized_coords(bbox1, image1)
        
        # Convert to bbox for image2
        bbox2 = normalized_coords_to_bbox(norm_bbox, image2)
        
        # Should represent same relative positions
        expected_bbox2 = (20, 20, 180, 180)  # 2x scaling
        assert bbox2 == expected_bbox2


class TestPatternMask:
    """Test pattern mask creation."""
    
    def test_create_pattern_mask_rectangle(self):
        """Test mask creation for rectangle pattern."""
        image = FibsemImage.generate_blank_image(resolution=[256, 256], pixel_size=Point(x=1e-9, y=1e-9))
        
        pattern = RectanglePattern(
            width=20e-6, height=20e-6,
            point=Point(x=0.0, y=0.0)
        )
        stage = FibsemMillingStage(name="Test Rectangle", pattern=pattern)
        
        mask = create_pattern_mask(stage, image)
        
        assert mask.shape == image.data.shape
        assert mask.dtype == bool
        # Should have some True values for the rectangle
        assert np.any(mask)
    
    def test_create_pattern_mask_circle(self):
        """Test mask creation for circle pattern."""
        image = FibsemImage.generate_blank_image(resolution=[256, 256], pixel_size=Point(x=1e-9, y=1e-9))
        
        pattern = CirclePattern(
            radius=10e-6,
            point=Point(x=0.0, y=0.0)
        )
        stage = FibsemMillingStage(name="Test Circle", pattern=pattern)
        
        mask = create_pattern_mask(stage, image)
        
        assert mask.shape == image.data.shape
        assert mask.dtype == bool
        # Should have some True values for the circle
        assert np.any(mask)
    
    def test_create_pattern_mask_exclusions_parameter(self):
        """Test mask creation with exclusions parameter."""
        image = FibsemImage.generate_blank_image(resolution=[256, 256], pixel_size=Point(x=1e-9, y=1e-9))
        
        # Create pattern with exclusion flag
        pattern = RectanglePattern(width=10e-6, height=10e-6)
        # Manually set exclusion on the first shape
        shapes = pattern.define()
        if shapes:
            shapes[0].is_exclusion = True
        
        stage = FibsemMillingStage(name="Test", pattern=pattern)
        
        # Test both include_exclusions values
        mask1 = create_pattern_mask(stage, image, include_exclusions=False)
        mask2 = create_pattern_mask(stage, image, include_exclusions=True)
        
        assert mask1.shape == image.data.shape
        assert mask2.shape == image.data.shape
        assert mask1.dtype == bool
        assert mask2.dtype == bool
    
    def test_create_pattern_mask_different_image_sizes(self):
        """Test that mask shape matches image shape for different sizes."""
        pattern = RectanglePattern(width=5e-6, height=5e-6)
        stage = FibsemMillingStage(name="Test", pattern=pattern)
        
        # Test with different image sizes
        for width, height in [(50, 50), (100, 200), (256, 128)]:
            image = FibsemImage.generate_blank_image(resolution=[width, height], pixel_size=Point(x=1e-9, y=1e-9))
            mask = create_pattern_mask(stage, image)
            assert mask.shape == (height, width)


class TestUtilityFunctionIntegration:
    """Test integration between utility functions."""
    
    def test_mask_and_bbox_integration(self):
        """Test that mask and bounding box functions work together consistently."""
        image = FibsemImage.generate_blank_image(resolution=[256, 256], pixel_size=Point(x=1e-9, y=1e-9))
        
        pattern = RectanglePattern(
            width=50e-6, height=30e-6,
            point=Point(x=20e-6, y=-10e-6)
        )
        stage = FibsemMillingStage(name="Test", pattern=pattern)
        
        # Get both mask and bounding box
        mask = create_pattern_mask(stage, image)
        bbox = get_pattern_bounding_box(stage, image)
        
        if bbox != (0, 0, 0, 0):
            # Bounding box should encompass all True pixels in mask
            mask_coords = np.where(mask)
            if len(mask_coords[0]) > 0:
                mask_y_min, mask_y_max = mask_coords[0].min(), mask_coords[0].max()
                mask_x_min, mask_x_max = mask_coords[1].min(), mask_coords[1].max()
                
                x_min, y_min, x_max, y_max = bbox
                
                # Bounding box should contain all mask pixels
                assert x_min <= mask_x_min
                assert y_min <= mask_y_min
                assert x_max >= mask_x_max
                assert y_max >= mask_y_max
    
    def test_bbox_expansion_bounds(self):
        """Test that bbox expansion respects image bounds."""
        image = FibsemImage.generate_blank_image(resolution=[100, 100], pixel_size=Point(x=1e-9, y=1e-9))
        
        # Create pattern near edge
        pattern = RectanglePattern(
            width=20e-6, height=20e-6,
            point=Point(x=40e-6, y=40e-6)  # Near edge
        )
        stage = FibsemMillingStage(name="Edge Test", pattern=pattern)
        
        # Large expansion should not exceed image bounds
        bbox = get_pattern_bounding_box(stage, image, expand_percent=1000.0)
        
        if bbox != (0, 0, 0, 0):
            x_min, y_min, x_max, y_max = bbox
            assert x_min >= 0
            assert y_min >= 0
            assert x_max < image.data.shape[1]
            assert y_max < image.data.shape[0]


if __name__ == "__main__":
    pytest.main([__file__])