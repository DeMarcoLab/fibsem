from __future__ import annotations
from dataclasses import fields

from typing import Type
import numpy as np
import pytest

from fibsem.milling.patterning.patterns2 import (
    BasePattern,
    CirclePattern,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemRectangleSettings,
    LinePattern,
    RectanglePattern,
    TrenchPattern,
    ArrayPattern,
    CloverPattern,
    FiducialPattern,
)
from fibsem.milling.patterning import (
    MILLING_PATTERNS,
    get_pattern,
)
from fibsem.structures import (
    CrossSectionPattern,
    Point,
)

@pytest.mark.parametrize("pattern", list(MILLING_PATTERNS.values()))
def test_required_attributes(pattern: Type[BasePattern]) -> None:
    # test that core attributes are not inherited by required attributes for specific patterns
    p = pattern()
    assert "point" not in p.required_attributes
    assert "shapes" not in p.required_attributes
    assert "name" not in p.required_attributes

    for f in fields(p):
        if f.name in ["point", "shapes", "name"]:
            continue
        # check that all fields are in required attributes
        assert f.name in p.required_attributes, (
            f"{f.name} not in required attributes for {p.name}"
)


def test_circle_settings():
    """
    Test the initialization, serialization, and deserialization of the FibsemCircleSettings class.
    This test verifies that:
    - The FibsemCircleSettings object is correctly initialized with the given parameters.
    - The to_dict() method accurately serializes the object's attributes to a dictionary.
    - The from_dict() method correctly reconstructs a FibsemCircleSettings object from a dictionary.
    - All attributes (radius, depth, centre_x, centre_y) retain their values throughout the process.
    """

    radius = 5e-6
    depth = 1e-6
    centre_x = 1e-6
    centre_y = -1e-6
    circle_settings = FibsemCircleSettings(
        radius=radius,
        depth=depth,
        centre_x=centre_x,
        centre_y=centre_y,
    )
    assert circle_settings.radius == radius
    assert circle_settings.depth == depth
    assert circle_settings.centre_x == centre_x
    assert circle_settings.centre_y == centre_y

    ddict = circle_settings.to_dict()
    assert ddict["radius"] == radius
    assert ddict["depth"] == depth
    assert ddict["centre_x"] == centre_x
    assert ddict["centre_y"] == centre_y

    circle_settings2 = FibsemCircleSettings.from_dict(ddict)
    assert circle_settings2.radius == radius
    assert circle_settings2.depth == depth
    assert circle_settings2.centre_x == centre_x
    assert circle_settings2.centre_y == centre_y


def test_line_settings():
    """
    Test the initialization, dictionary conversion, and reconstruction of a FibsemLineSettings object.

    This test verifies that:
    - The FibsemLineSettings object is correctly initialized with the provided start and end coordinates and depth.
    - The object's `to_dict` method returns a dictionary with the correct values.
    - The `from_dict` method reconstructs an equivalent FibsemLineSettings object from the dictionary.
    """
    start_x = 0
    start_y = 0
    end_x = 2e-6
    end_y = 2e-6
    depth = 1e-6

    line_settings = FibsemLineSettings(
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        depth=depth,
    )
    assert line_settings.start_x == start_x
    assert line_settings.start_y == start_y
    assert line_settings.end_x == end_x
    assert line_settings.end_y == end_y
    assert line_settings.depth == depth

    ddict = line_settings.to_dict()
    assert ddict["start_x"] == start_x
    assert ddict["start_y"] == start_y
    assert ddict["end_x"] == end_x
    assert ddict["end_y"] == end_y
    assert ddict["depth"] == depth

    line_settings2 = FibsemLineSettings.from_dict(ddict)
    assert line_settings2.start_x == start_x
    assert line_settings2.start_y == start_y
    assert line_settings2.end_x == end_x
    assert line_settings2.end_y == end_y
    assert line_settings2.depth == depth

def test_rectangle_settings():
    """Test the initialization, dictionary conversion, and reconstruction of a FibsemRectangleSettings object."""
    centre_x = 0
    centre_y = 0
    width = 2e-6
    height = 3e-6
    depth = 1e-6
    rotation = 0

    rect_settings = FibsemRectangleSettings(
        centre_x=centre_x,
        centre_y=centre_y,
        width=width,
        height=height,
        depth=depth,
        rotation=rotation,
    )
    assert rect_settings.centre_x == centre_x
    assert rect_settings.centre_y == centre_y
    assert rect_settings.width == width
    assert rect_settings.height == height
    assert rect_settings.depth == depth
    assert rect_settings.rotation == rotation

    ddict = rect_settings.to_dict()
    assert ddict["centre_x"] == centre_x
    assert ddict["centre_y"] == centre_y
    assert ddict["width"] == width
    assert ddict["height"] == height
    assert ddict["depth"] == depth
    assert ddict["rotation"] == rotation

    rect_settings2 = FibsemRectangleSettings.from_dict(ddict)
    assert rect_settings2.centre_x == centre_x
    assert rect_settings2.centre_y == centre_y
    assert rect_settings2.width == width
    assert rect_settings2.height == height
    assert rect_settings2.depth == depth
    assert rect_settings2.rotation == rotation


def test_circle_pattern():
    """
    Test the CirclePattern class for correct initialization, shape definition, 
    dictionary serialization/deserialization, and attribute consistency.
    This test verifies that:
    - A CirclePattern object can be created with specified radius and depth.
    - The define() method correctly creates a shape with the expected attributes.
    - The to_dict() method serializes the object with correct values.
    - The from_dict() method deserializes the object and preserves the original attributes.
    """

    radius = 5e-6
    depth = 1e-6
    circle = CirclePattern(
        radius=radius,
        depth=depth,
    )
    circle.define()
    assert len(circle.shapes) == 1
    assert circle.shapes[0].radius == radius
    assert circle.shapes[0].depth == depth
    assert circle.shapes[0].centre_x == 0
    assert circle.shapes[0].centre_y == 0

    ddict = circle.to_dict()
    assert ddict["radius"] == radius
    assert ddict["depth"] == depth
    assert ddict["point"]["x"] == 0
    assert ddict["point"]["y"] == 0

    circle2 = CirclePattern.from_dict(ddict)
    assert circle2.radius == radius
    assert circle2.depth == depth


def test_line_pattern():
    """
    Test the functionality of the LinePattern class.
    This test verifies that:
    - A LinePattern object can be created with specified start and end coordinates and depth.
    - The define() method correctly populates the shapes attribute.
    - The shape's attributes match the input parameters.
    - The to_dict() method serializes the object correctly.
    - The from_dict() method deserializes the object correctly and preserves all attributes.
    """

    start_x = 0
    start_y = 0
    end_x = 2e-6
    end_y = 2e-6
    depth = 1e-6
    line = LinePattern(
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        depth=depth,
    )
    line.define()
    assert len(line.shapes) == 1
    assert line.shapes[0].start_x == start_x
    assert line.shapes[0].start_y == start_y
    assert line.shapes[0].end_x == end_x
    assert line.shapes[0].end_y == end_y
    assert line.shapes[0].depth == depth

    ddict = line.to_dict()
    assert ddict["start_x"] == start_x
    assert ddict["start_y"] == start_y
    assert ddict["end_x"] == end_x
    assert ddict["end_y"] == end_y
    assert ddict["depth"] == depth

    line2 = LinePattern.from_dict(ddict)
    assert line2.start_x == start_x
    assert line2.start_y == start_y
    assert line2.end_x == end_x
    assert line2.end_y == end_y
    assert line2.depth == depth


def test_rectangle_pattern():
    """
    Test the RectanglePattern class for correct initialization, shape definition,
    dictionary serialization/deserialization, and attribute consistency.
    This test verifies that:
    - A RectanglePattern object can be created with specified width, height, depth, and rotation.
    - The define() method correctly creates a shape with the expected attributes.
    - The to_dict() method serializes the object with correct values.
    - The from_dict() method deserializes the object and preserves the original attributes.
    """
    width = 10e-6
    height = 5e-6
    depth = 1e-6
    rotation = 0

    rect = RectanglePattern(
        width=width,
        height=height,
        depth=depth,
    )
    rect.define()

    # test shape definition
    assert len(rect.shapes) == 1
    assert rect.shapes[0].width == width
    assert rect.shapes[0].height == height
    assert rect.shapes[0].depth == depth
    assert rect.shapes[0].rotation == rotation
    assert rect.shapes[0].centre_x == 0
    assert rect.shapes[0].centre_y == 0


    # test serialization
    ddict = rect.to_dict()
    assert ddict["width"] == width
    assert ddict["height"] == height
    assert ddict["depth"] == depth
    assert ddict["rotation"] == rotation
    assert ddict["point"]["x"] == 0
    assert ddict["point"]["y"] == 0

    # test deserialization
    rect2 = RectanglePattern.from_dict(ddict)
    assert rect2.width == width
    assert rect2.height == height
    assert rect2.depth == depth
    assert rect2.rotation == rotation




class TestTrenchPattern:
    def test_init(self):
        # Test default initialization
        trench = TrenchPattern(
            width=100.0,
            depth=50.0,
            spacing=20.0,
            upper_trench_height=30.0,
            lower_trench_height=25.0
        )

        assert trench.width == 100.0
        assert trench.depth == 50.0
        assert trench.spacing == 20.0
        assert trench.upper_trench_height == 30.0
        assert trench.lower_trench_height == 25.0
        assert trench.cross_section == CrossSectionPattern.Rectangle
        assert trench.time == 0.0
        assert trench.fillet == 0.0
        assert trench.point.x == 0.0
        assert trench.point.y == 0.0
        assert trench.name == "Trench"
        assert trench.shapes is None

    def test_define_without_fillet(self):
        trench = TrenchPattern(
            width=100.0,
            depth=50.0,
            spacing=20.0,
            upper_trench_height=30.0,
            lower_trench_height=25.0,
            point=Point(10.0, 20.0),
            time=5.0
        )

        shapes = trench.define()
        
        assert len(shapes) == 2  # Should have upper and lower trench only
        assert isinstance(shapes[0], FibsemRectangleSettings)
        assert isinstance(shapes[1], FibsemRectangleSettings)
        
        # Check lower trench
        assert shapes[0].width == 100.0
        assert shapes[0].height == 25.0
        assert shapes[0].depth == 50.0
        assert shapes[0].centre_x == 10.0
        assert shapes[0].scan_direction == "BottomToTop"
        assert shapes[0].time == 5.0
        
        # Check upper trench
        assert shapes[1].width == 100.0
        assert shapes[1].height == 30.0
        assert shapes[1].depth == 50.0
        assert shapes[1].centre_x == 10.0
        assert shapes[1].scan_direction == "TopToBottom"
        assert shapes[1].time == 5.0
        
        # Check calculated positions
        expected_lower_y = 20.0 - (20.0 / 2 + 25.0 / 2)
        expected_upper_y = 20.0 + (20.0 / 2 + 30.0 / 2)
        assert abs(shapes[0].centre_y - expected_lower_y) < 1e-6
        assert abs(shapes[1].centre_y - expected_upper_y) < 1e-6

    def test_define_with_fillet(self):
        trench = TrenchPattern(
            width=100.0,
            depth=50.0,
            spacing=20.0,
            upper_trench_height=30.0,
            lower_trench_height=25.0,
            point=Point(10.0, 20.0),
            fillet=5.0
        )

        shapes = trench.define()
        
        assert len(shapes) == 10  # 2 trenches + 8 fillet shapes
        
        # First two shapes should be main trenches
        assert isinstance(shapes[0], FibsemRectangleSettings)
        assert isinstance(shapes[1], FibsemRectangleSettings)
        
        # Check width reduction due to fillet
        assert shapes[0].width == 90.0  # 100 - 2*5
        assert shapes[1].width == 90.0  # 100 - 2*5
        
        # Check fillet shapes
        fillet_shapes = shapes[2:]
        circle_shapes = [s for s in fillet_shapes if isinstance(s, FibsemCircleSettings)]
        rect_shapes = [s for s in fillet_shapes if isinstance(s, FibsemRectangleSettings)]
        
        assert len(circle_shapes) == 4  # 4 circle fillets
        assert len(rect_shapes) == 4  # 4 rectangle fills
        
        # Check fillet radius
        for circle in circle_shapes:
            assert circle.radius == 5.0

    def test_fillet_clipping(self):
        # Test when fillet is too large (more than half the upper trench height)
        trench = TrenchPattern(
            width=100.0,
            depth=50.0,
            spacing=20.0,
            upper_trench_height=30.0,
            lower_trench_height=25.0,
            point=Point(10.0, 20.0),
            fillet=20.0  # This is more than upper_trench_height/2
        )

        shapes = trench.define()
        
        # Check that fillet was clipped to upper_trench_height/2 = 15.0
        circle_shapes = [s for s in shapes if isinstance(s, FibsemCircleSettings)]
        for circle in circle_shapes:
            assert circle.radius == 15.0
            
        # Width should be reduced by 2*15.0
        assert shapes[0].width == 70.0  # 100 - 2*15

    def test_to_dict(self):
        trench = TrenchPattern(
            width=100.0,
            depth=50.0,
            spacing=20.0,
            upper_trench_height=30.0,
            lower_trench_height=25.0,
            point=Point(10.0, 20.0),
            time=5.0,
            fillet=5.0,
            cross_section=CrossSectionPattern.Rectangle
        )
        
        result_dict = trench.to_dict()
        
        assert result_dict["name"] == "Trench"
        assert result_dict["width"] == 100.0
        assert result_dict["depth"] == 50.0
        assert result_dict["spacing"] == 20.0
        assert result_dict["upper_trench_height"] == 30.0
        assert result_dict["lower_trench_height"] == 25.0
        assert result_dict["point"]["x"] == 10.0
        assert result_dict["point"]["y"] == 20.0
        assert result_dict["time"] == 5.0
        assert result_dict["fillet"] == 5.0
        assert result_dict["cross_section"] == "Rectangle"

    def test_from_dict(self):
        test_dict = {
            "width": 100.0,
            "depth": 50.0,
            "spacing": 20.0,
            "upper_trench_height": 30.0,
            "lower_trench_height": 25.0,
            "point": {"x": 10.0, "y": 20.0},
            "time": 5.0,
            "fillet": 5.0,
            "cross_section": "Rectangle"
        }
        
        trench = TrenchPattern.from_dict(test_dict)
        
        assert trench.width == 100.0
        assert trench.depth == 50.0
        assert trench.spacing == 20.0
        assert trench.upper_trench_height == 30.0
        assert trench.lower_trench_height == 25.0
        assert trench.point.x == 10.0
        assert trench.point.y == 20.0
        assert trench.time == 5.0
        assert trench.fillet == 5.0
        assert trench.cross_section == CrossSectionPattern.Rectangle

    def test_from_dict_default_values(self):
        # Test with minimal dict and default values
        test_dict = {
            "width": 100.0,
            "depth": 50.0,
            "spacing": 20.0,
            "upper_trench_height": 30.0,
            "lower_trench_height": 25.0,
        }
        
        trench = TrenchPattern.from_dict(test_dict)
        
        assert trench.width == 100.0
        assert trench.fillet == 0.0
        assert trench.time == 0.0
        assert trench.cross_section == CrossSectionPattern.Rectangle
        assert trench.point.x == 0.0
        assert trench.point.y == 0.0





class TestArrayPattern:
    def test_init(self):
        # Test default initialization
        array = ArrayPattern(
            width=10.0,
            height=20.0,
            depth=30.0,
            n_columns=3,
            n_rows=2,
            pitch_vertical=50.0,
            pitch_horizontal=40.0
        )

        assert array.width == 10.0
        assert array.height == 20.0
        assert array.depth == 30.0
        assert array.n_columns == 3
        assert array.n_rows == 2
        assert array.pitch_vertical == 50.0
        assert array.pitch_horizontal == 40.0
        assert array.passes == 0
        assert array.rotation == 0
        assert array.scan_direction == "TopToBottom"
        assert array.cross_section == CrossSectionPattern.Rectangle
        assert array.point.x == 0.0
        assert array.point.y == 0.0
        assert array.name == "ArrayPattern"
        assert array.shapes is None

    def test_define(self):
        # Test with 2x2 array
        array = ArrayPattern(
            width=10.0,
            height=20.0,
            depth=30.0,
            n_columns=2,
            n_rows=2,
            pitch_vertical=50.0,
            pitch_horizontal=40.0,
            passes=2,
            rotation=45,
            scan_direction="LeftToRight",
            point=Point(5.0, 15.0)
        )

        shapes = array.define()
        
        assert len(shapes) == 4  # 2x2 array = 4 shapes
        
        for shape in shapes:
            assert isinstance(shape, FibsemRectangleSettings)
            assert shape.width == 10.0
            assert shape.height == 20.0
            assert shape.depth == 30.0
            assert shape.passes == 2
            assert shape.rotation == 45
            assert shape.scan_direction == "LeftToRight"
            assert shape.cross_section == CrossSectionPattern.Rectangle
        
        # Check positions - should form a 2x2 grid centered at point
        expected_positions = [
            (5.0 - 40.0/2, 15.0 - 50.0/2),  # bottom left
            (5.0 - 40.0/2, 15.0 + 50.0/2),  # top left
            (5.0 + 40.0/2, 15.0 - 50.0/2),  # bottom right
            (5.0 + 40.0/2, 15.0 + 50.0/2),  # top right
        ]
        
        for shape in shapes:
            pos = (shape.centre_x, shape.centre_y)
            assert pos in expected_positions

    def test_to_dict(self):
        array = ArrayPattern(
            width=10.0,
            height=20.0,
            depth=30.0,
            n_columns=3,
            n_rows=2,
            pitch_vertical=50.0,
            pitch_horizontal=40.0,
            passes=2,
            rotation=45,
            scan_direction="LeftToRight",
            cross_section=CrossSectionPattern.Rectangle,
            point=Point(5.0, 15.0)
        )
        
        result_dict = array.to_dict()
        
        assert result_dict["name"] == "ArrayPattern"
        assert result_dict["width"] == 10.0
        assert result_dict["height"] == 20.0
        assert result_dict["depth"] == 30.0
        assert result_dict["n_columns"] == 3
        assert result_dict["n_rows"] == 2
        assert result_dict["pitch_vertical"] == 50.0
        assert result_dict["pitch_horizontal"] == 40.0
        assert result_dict["passes"] == 2
        assert result_dict["rotation"] == 45
        assert result_dict["scan_direction"] == "LeftToRight"
        assert result_dict["cross_section"] == "Rectangle"
        assert result_dict["point"]["x"] == 5.0
        assert result_dict["point"]["y"] == 15.0

    def test_from_dict(self):
        test_dict = {
            "width": 10.0,
            "height": 20.0,
            "depth": 30.0,
            "n_columns": 3,
            "n_rows": 2,
            "pitch_vertical": 50.0,
            "pitch_horizontal": 40.0,
            "passes": 2,
            "rotation": 45,
            "scan_direction": "LeftToRight",
            "cross_section": "Rectangle",
            "point": {"x": 5.0, "y": 15.0}
        }
        
        array = ArrayPattern.from_dict(test_dict)
        
        assert array.width == 10.0
        assert array.height == 20.0
        assert array.depth == 30.0
        assert array.n_columns == 3
        assert array.n_rows == 2
        assert array.pitch_vertical == 50.0
        assert array.pitch_horizontal == 40.0
        assert array.passes == 2
        assert array.rotation == 45
        assert array.scan_direction == "LeftToRight"
        assert array.cross_section == CrossSectionPattern.Rectangle
        assert array.point.x == 5.0
        assert array.point.y == 15.0

    def test_from_dict_default_values(self):
        # Test with minimal dict and default values
        test_dict = {
            "width": 10.0,
            "height": 20.0,
            "depth": 30.0,
            "n_columns": 3,
            "n_rows": 2,
            "pitch_vertical": 50.0,
            "pitch_horizontal": 40.0,
        }
        
        array = ArrayPattern.from_dict(test_dict)
        
        assert array.passes == 0
        assert array.rotation == 0
        assert array.scan_direction == "TopToBottom"
        assert array.cross_section == CrossSectionPattern.Rectangle
        assert array.point.x == 0.0
        assert array.point.y == 0.0


class TestCloverPattern:
    def test_init(self):
        # Test initialization
        clover = CloverPattern(
            radius=10.0,
            depth=5.0
        )
        
        assert clover.radius == 10.0
        assert clover.depth == 5.0
        assert clover.name == "Clover"
        assert clover.point.x == 0.0
        assert clover.point.y == 0.0
        assert clover.shapes is None

    def test_define(self):
        clover = CloverPattern(
            radius=10.0,
            depth=5.0,
            point=Point(5.0, 15.0)
        )
        
        shapes = clover.define()
        
        assert len(shapes) == 4  # 3 circles + 1 rectangle
        
        # Check the circles
        circles = [s for s in shapes if isinstance(s, FibsemCircleSettings)]
        assert len(circles) == 3
        
        for circle in circles:
            assert circle.radius == 10.0
            assert circle.depth == 5.0
        
        # Check positions
        expected_circle_positions = [
            (5.0, 25.0),  # top
            (15.0, 15.0),  # right
            (-5.0, 15.0),  # left
        ]
        
        for circle in circles:
            pos = (circle.centre_x, circle.centre_y)
            assert pos in expected_circle_positions
        
        # Check the stem (rectangle)
        stem = [s for s in shapes if isinstance(s, FibsemRectangleSettings)][0]
        assert stem.width == 10.0 / 4  # radius / 4
        assert stem.height == 10.0 * 2  # radius * 2
        assert stem.depth == 5.0
        assert stem.centre_x == 5.0
        assert stem.centre_y == 5.0  # point.y - radius
        assert stem.scan_direction == "TopToBottom"

    def test_to_dict(self):
        clover = CloverPattern(
            radius=10.0,
            depth=5.0,
            point=Point(5.0, 15.0)
        )
        
        result_dict = clover.to_dict()
        
        assert result_dict["name"] == "Clover"
        assert result_dict["radius"] == 10.0
        assert result_dict["depth"] == 5.0
        assert result_dict["point"]["x"] == 5.0
        assert result_dict["point"]["y"] == 15.0

    def test_from_dict(self):
        test_dict = {
            "radius": 10.0,
            "depth": 5.0,
            "point": {"x": 5.0, "y": 15.0}
        }
        
        clover = CloverPattern.from_dict(test_dict)
        
        assert clover.radius == 10.0
        assert clover.depth == 5.0
        assert clover.point.x == 5.0
        assert clover.point.y == 15.0


class TestFiducialPattern:
    def test_init(self):
        # Test initialization
        fiducial = FiducialPattern(
            width=10.0,
            height=20.0,
            depth=5.0
        )
        
        assert fiducial.width == 10.0
        assert fiducial.height == 20.0
        assert fiducial.depth == 5.0
        assert fiducial.rotation == 45
        assert fiducial.cross_section == CrossSectionPattern.Rectangle
        assert fiducial.name == "Fiducial"
        assert fiducial.point.x == 0.0
        assert fiducial.point.y == 0.0
        assert fiducial.shapes is None

    def test_define(self):
        fiducial = FiducialPattern(
            width=10.0,
            height=20.0,
            depth=5.0,
            rotation=45,
            cross_section=CrossSectionPattern.Rectangle,
            point=Point(5.0, 15.0)
        )
        
        shapes = fiducial.define()
        
        assert len(shapes) == 2  # Cross shape has 2 rectangles
        
        for shape in shapes:
            assert isinstance(shape, FibsemRectangleSettings)
            assert shape.width == 10.0
            assert shape.height == 20.0
            assert shape.depth == 5.0
            assert shape.centre_x == 5.0
            assert shape.centre_y == 15.0
            assert shape.scan_direction == "TopToBottom"
            assert shape.cross_section == CrossSectionPattern.Rectangle
        
        # First rectangle should be rotated by rotation value
        assert shapes[0].rotation == 45 * (np.pi / 180)
        
        # Second rectangle should be rotated 90 degrees more
        assert shapes[1].rotation == (45 + 90) * (np.pi / 180)

    def test_to_dict(self):
        fiducial = FiducialPattern(
            width=10.0,
            height=20.0,
            depth=5.0,
            rotation=45,
            cross_section=CrossSectionPattern.Rectangle,
            point=Point(5.0, 15.0)
        )
        
        result_dict = fiducial.to_dict()
        
        assert result_dict["name"] == "Fiducial"
        assert result_dict["width"] == 10.0
        assert result_dict["height"] == 20.0
        assert result_dict["depth"] == 5.0
        assert result_dict["rotation"] == 45
        assert result_dict["cross_section"] == "Rectangle"
        assert result_dict["point"]["x"] == 5.0
        assert result_dict["point"]["y"] == 15.0

    def test_from_dict(self):
        test_dict = {
            "width": 10.0,
            "height": 20.0,
            "depth": 5.0,
            "rotation": 45,
            "cross_section": "Rectangle",
            "point": {"x": 5.0, "y": 15.0}
        }
        
        fiducial = FiducialPattern.from_dict(test_dict)
        
        assert fiducial.width == 10.0
        assert fiducial.height == 20.0
        assert fiducial.depth == 5.0
        assert fiducial.rotation == 45
        assert fiducial.cross_section == CrossSectionPattern.Rectangle
        assert fiducial.point.x == 5.0
        assert fiducial.point.y == 15.0

    def test_from_dict_default_values(self):
        # Test with minimal dict and default values
        test_dict = {
            "width": 10.0,
            "height": 20.0,
            "depth": 5.0
        }
        
        fiducial = FiducialPattern.from_dict(test_dict)
        
        assert fiducial.rotation == 45
        assert fiducial.cross_section == CrossSectionPattern.Rectangle
        assert fiducial.point.x == 0.0
        assert fiducial.point.y == 0.0


class TestGetPattern:
    def test_get_pattern(self):
        # Test that get_pattern correctly instantiates a pattern
        config = {
            "width": 10.0,
            "height": 20.0,
            "depth": 5.0
        }
        
        pattern = get_pattern("rectangle", config)
        
        assert isinstance(pattern, RectanglePattern)
        assert pattern.width == 10.0
        assert pattern.height == 20.0
        assert pattern.depth == 5.0
        
        # Test with another pattern type
        config = {
            "radius": 10.0,
            "depth": 5.0
        }
        
        pattern = get_pattern("circle", config)
        
        assert isinstance(pattern, CirclePattern)
        assert pattern.radius == 10.0
        assert pattern.depth == 5.0
        
        # Test case insensitivity
        pattern = get_pattern("CIRCLE", config)
        assert isinstance(pattern, CirclePattern)
        
        # Verify all registered patterns can be retrieved
        for pattern_name in MILLING_PATTERNS.keys():
            # Create a minimal config with required attributes
            cls = MILLING_PATTERNS[pattern_name]
            required_attrs = cls.__init__.__annotations__
            
            # Create a basic config with sample values for required attributes
            minimal_config = {}
            for attr, t in required_attrs.items():
                if attr != 'return' and attr not in ['point', 'shapes', 'name']:
                    if t == float:
                        minimal_config[attr] = 10.0
                    elif t == int:
                        minimal_config[attr] = 3
                    elif t == bool:
                        minimal_config[attr] = False
                    elif t == str:
                        minimal_config[attr] = "TopToBottom"
            
            pattern = get_pattern(pattern_name, minimal_config)
            assert isinstance(pattern, MILLING_PATTERNS[pattern_name])