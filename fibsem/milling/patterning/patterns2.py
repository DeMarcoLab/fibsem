from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Dict, List, Tuple, Union

import numpy as np

from fibsem import constants
from fibsem.structures import (
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemPatternSettings,
    FibsemRectangleSettings,
    Point,
)

# TODO: define the configuration for each key,
# e.g. 
# "width": {"type": "float", "min": 0, "max": 1000, "default": 100, "description": "Width of the rectangle"}
# "cross_section": {"type": "str", "options": [cs.name for cs in CrossSectionPattern], "default": "Rectangle", "description": "Cross section of the milling pattern"}

DEFAULT_POINT_DDICT = {"x": 0.0, "y": 0.0}

####### Combo Patterns

CORE_PATTERN_ATTRIBUTES = ["name", "point", "shapes"]

@dataclass
class BasePattern(ABC):
    # name: str = "BasePattern"
    # point: Point = Point() 
    # shapes: List[FibsemPatternSettings] = None
    # TODO: investigate TypeError: non-default argument 'width' follows default argument when uncommenting the above lines

    @abstractmethod
    def define(self) -> List[FibsemPatternSettings]:
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, ddict: dict) -> "BasePattern":
        pass

    @property
    def required_attributes(self) -> Tuple[str]:
        return [field.name for field in fields(self) if field.name not in CORE_PATTERN_ATTRIBUTES]
    
    @property
    def advanced_attributes(self) -> List[str]:
        if hasattr(self, "_advanced_attributes"):
            return self._advanced_attributes
        return []

    @property
    def volume(self) -> float:
        # calculate the total volume of the milling pattern (sum of all shapes)
        return sum([shape.volume for shape in self.define()])

@dataclass
class BitmapPattern(BasePattern):
    width: float
    height: float
    depth: float
    rotation: float = 0
    path: str = ""
    shapes: List[FibsemPatternSettings] = None
    point: Point = Point()
    name: str = "Bitmap"

    def define(self) -> List[FibsemPatternSettings]:

        shape = FibsemBitmapSettings(
            width=self.width,
            height=self.height,
            depth=self.depth,
            rotation=self.rotation,
            path=self.path,
            centre_x=self.point.x,
            centre_y=self.point.y,
        )

        self.shapes = [shape]
        return self.shapes
    
    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "rotation": self.rotation,
            "path": self.path
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "BitmapPattern":
        return cls(
            width=ddict["width"],
            height=ddict["height"],
            depth=ddict["depth"],
            rotation=ddict.get("rotation", 0),
            path=ddict.get("path", ""),
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )


@dataclass
class RectanglePattern(BasePattern):
    width: float
    height: float
    depth: float
    rotation: float = 0
    time: float = 0
    passes: int = 0
    scan_direction: str = "TopToBottom"
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle
    point: Point = Point()
    shapes: List[FibsemPatternSettings] = None
    name: str = "Rectangle"
    _advanced_attributes = ["time", "passes"] # TODO: add for other patterns

    def define(self) -> List[FibsemRectangleSettings]:

        shape = FibsemRectangleSettings(
            width=self.width,
            height=self.height,
            depth=self.depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            rotation=self.rotation * constants.DEGREES_TO_RADIANS,
            passes=self.passes,
            time=self.time,
            scan_direction=self.scan_direction,
            cross_section=self.cross_section,
        )

        self.shapes = [shape]
        return self.shapes
    
    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "rotation": self.rotation,
            "time": self.time,
            "passes": self.passes,
            "scan_direction": self.scan_direction,
            "cross_section": self.cross_section.name
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "RectanglePattern":
        return cls(
            width=ddict["width"],
            height=ddict["height"],
            depth=ddict["depth"],
            rotation=ddict.get("rotation", 0),
            time=ddict.get("time", 0),
            passes=ddict.get("passes", 0),
            scan_direction=ddict.get("scan_direction", "TopToBottom"),
            cross_section=CrossSectionPattern[ddict.get("cross_section", "Rectangle")],
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )


@dataclass
class LinePattern(BasePattern):
    start_x: float
    end_x: float
    start_y: float
    end_y: float
    depth: float
    point: Point = Point()
    shapes: List[FibsemPatternSettings] = None
    name: str = "Line"

    def define(self) -> List[FibsemLineSettings]:
        shape = FibsemLineSettings(
            start_x=self.start_x,
            end_x=self.end_x,
            start_y=self.start_y,
            end_y=self.end_y,
            depth=self.depth,
        )
        self.shapes = [shape]
        return self.shapes
    
    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "start_x": self.start_x,
            "end_x": self.end_x,
            "start_y": self.start_y,
            "end_y": self.end_y,
            "depth": self.depth
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "LinePattern":
        return cls(
            start_x=ddict["start_x"],
            end_x=ddict["end_x"],
            start_y=ddict["start_y"],
            end_y=ddict["end_y"],
            depth=ddict["depth"],
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )

@dataclass
class CirclePattern(BasePattern):
    radius: float
    depth: float
    thickness: float = 0
    name: str = "Circle"
    shapes: List[FibsemPatternSettings] = None
    point: Point = Point()

    def define(self) -> List[FibsemCircleSettings]:
        
        shape = FibsemCircleSettings(
            radius=self.radius,
            depth=self.depth,
            thickness=self.thickness,
            centre_x=self.point.x,
            centre_y=self.point.y,
        )
        self.shapes = [shape]
        return self.shapes
    
    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "radius": self.radius,
            "depth": self.depth,
            "thickness": self.thickness
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "CirclePattern":
        return cls(
            radius=ddict["radius"],
            depth=ddict["depth"],
            thickness=ddict.get("thickness", 0),
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )

@dataclass
class TrenchPattern(BasePattern):
    width: float
    depth: float
    spacing: float
    upper_trench_height: float
    lower_trench_height: float
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle
    time: float = 0.0
    point: Point = Point()
    name: str = "Trench"
    shapes: List[FibsemPatternSettings] = None
    _advanced_attributes = ["time"]

    def define(self) -> List[FibsemRectangleSettings]:

        point = self.point
        width = self.width
        spacing = self.spacing
        upper_trench_height = self.upper_trench_height
        lower_trench_height = self.lower_trench_height
        depth = self.depth
        cross_section = self.cross_section
        time = self.time

        # calculate the centre of the upper and lower trench
        centre_lower_y = point.y - (spacing / 2 + lower_trench_height / 2)
        centre_upper_y = point.y + (spacing / 2 + upper_trench_height / 2)

        # mill settings
        lower_trench_settings = FibsemRectangleSettings(
            width=width,
            height=lower_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_lower_y,
            scan_direction="BottomToTop",
            cross_section = cross_section,
            time = time
        )

        upper_trench_settings = FibsemRectangleSettings(
            width=width,
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_upper_y,
            scan_direction="TopToBottom",
            cross_section = cross_section,
            time = time
        )

        self.shapes = [lower_trench_settings, upper_trench_settings]
        return self.shapes

    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "width": self.width,
            "depth": self.depth,
            "spacing": self.spacing,
            "upper_trench_height": self.upper_trench_height,
            "lower_trench_height": self.lower_trench_height,
            "cross_section": self.cross_section.name,
            "time": self.time
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "TrenchPattern":
        return cls(
            width=ddict["width"],
            depth=ddict["depth"],
            spacing=ddict["spacing"],
            upper_trench_height=ddict["upper_trench_height"],
            lower_trench_height=ddict["lower_trench_height"],
            cross_section=CrossSectionPattern[ddict.get("cross_section", "Rectangle")],
            time=ddict.get("time", 0),
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )

@dataclass
class HorseshoePattern(BasePattern):
    width: float
    upper_trench_height: float
    lower_trench_height: float
    spacing: float
    depth: float
    side_width: float
    inverted: bool = False
    scan_direction: str = "TopToBottom"
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle
    shapes: List[FibsemPatternSettings] = None
    point: Point = Point()
    name: str = "Horseshoe"
    # ref: "horseshoe" terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14

    def define(self) -> List[FibsemRectangleSettings]:
        """Calculate the trench milling patterns"""

        point = self.point
        width = self.width
        depth = self.depth
        spacing = self.spacing
        lower_trench_height = self.lower_trench_height
        upper_trench_height = self.upper_trench_height
        cross_section = self.cross_section
        side_width = self.side_width

        # calculate the centre of the upper and lower trench
        centre_upper_y = point.y + (spacing / 2 + upper_trench_height / 2)
        centre_lower_y = point.y - (spacing / 2 + lower_trench_height / 2)

        # calculate the centre of the side trench
        side_height = spacing + upper_trench_height + lower_trench_height
        side_offset = (width / 2) + (side_width / 2)
        if self.inverted:
            side_offset = -side_offset
        side_x = point.x + side_offset
        # to account for assymetric trench heights
        side_y = point.y + (upper_trench_height - lower_trench_height) / 2

        lower_pattern = FibsemRectangleSettings(
            width=width,
            height=lower_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_lower_y,
            scan_direction="BottomToTop",
            cross_section = cross_section
        )

        upper_pattern = FibsemRectangleSettings(
            width=width,
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_upper_y,
            scan_direction="TopToBottom",
            cross_section = cross_section
        )

        side_pattern = FibsemRectangleSettings(
            width=side_width,
            height=side_height,
            depth=depth,
            centre_x=side_x,
            centre_y=side_y,
            scan_direction=self.scan_direction,
            cross_section=cross_section
        )

        self.shapes = [lower_pattern, upper_pattern, side_pattern]
        return self.shapes
    
    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "width": self.width,
            "spacing": self.spacing,
            "depth": self.depth,
            "upper_trench_height": self.upper_trench_height,
            "lower_trench_height": self.lower_trench_height,
            "side_width": self.side_width,
            "inverted": self.inverted,
            "scan_direction": self.scan_direction,
            "cross_section": self.cross_section.name
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "HorseshoePattern":
        return cls(
            width=ddict["width"],
            spacing=ddict["spacing"],
            depth=ddict["depth"],
            upper_trench_height=ddict["upper_trench_height"],
            lower_trench_height=ddict["lower_trench_height"],
            side_width=ddict["side_width"],
            inverted=ddict.get("inverted", False),
            scan_direction=ddict.get("scan_direction", "TopToBottom"),
            cross_section=CrossSectionPattern[ddict.get("cross_section", "Rectangle")],
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )
    

@dataclass
class HorseshoePatternVertical(BasePattern):
    width: float
    height: float
    side_trench_width: float
    top_trench_height: float
    depth: float
    scan_direction: str = "TopToBottom"
    inverted: bool = False
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle
    point: Point = Point()
    name: str = "HorseshoeVertical"
    # ref: "horseshoe" terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14

    def define(self) -> List[FibsemRectangleSettings]:
        """Calculate the horseshoe vertical milling patterns"""

        point = self.point
        width = self.width
        height = self.height
        trench_width = self.side_trench_width
        upper_trench_height = self.top_trench_height
        depth = self.depth
        scan_direction = self.scan_direction
        inverted = self.inverted
        cross_section = self.cross_section

        left_pattern = FibsemRectangleSettings(
            width=trench_width,
            height=height,
            depth=depth,
            centre_x=point.x - (width / 2) - (trench_width / 2),
            centre_y=point.y,
            scan_direction="LeftToRight",
            cross_section=cross_section
        )

        right_pattern = FibsemRectangleSettings(
            width=trench_width,
            height=height,
            depth=depth,
            centre_x=point.x + (width / 2) + (trench_width / 2),
            centre_y=point.y,
            scan_direction="RightToLeft",
            cross_section=cross_section
        )
        y_offset = (height / 2) + (upper_trench_height / 2)
        if inverted:
            y_offset = -y_offset
        upper_pattern = FibsemRectangleSettings(
            width=width + (2 * trench_width),
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y + y_offset,
            scan_direction=scan_direction,
            cross_section=cross_section
        )
        
        self.shapes = [left_pattern,right_pattern, upper_pattern]
        return self.shapes
    
    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "width": self.width,
            "height": self.height,
            "side_trench_width": self.side_trench_width,
            "top_trench_height": self.top_trench_height,
            "depth": self.depth,
            "scan_direction": self.scan_direction,
            "inverted": self.inverted,
            "cross_section": self.cross_section.name
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "HorseshoePatternVertical":
        return cls(
            width=ddict["width"],
            height=ddict["height"],
            side_trench_width=ddict["side_trench_width"],
            top_trench_height=ddict["top_trench_height"],
            depth=ddict["depth"],
            scan_direction=ddict.get("scan_direction", "TopToBottom"),
            inverted=ddict.get("inverted", False),
            cross_section=CrossSectionPattern[ddict.get("cross_section", "Rectangle")],
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )

@dataclass
class SerialSectionPattern(BasePattern):
    section_thickness: float
    section_width: float
    section_depth: float
    side_width: float
    side_height: float = 0
    side_depth: float = 0
    inverted: bool = False
    use_side_patterns: bool = True
    name: str = "SerialSection"
    point: Point = Point()
    shapes: List[FibsemPatternSettings] = None
    # ref: "serial-liftout section" https://www.nature.com/articles/s41592-023-02113-5

    def define(self) -> List[FibsemRectangleSettings]:
        """Calculate the serial liftout sectioning milling patterns"""

        point = self.point
        section_thickness = self.section_thickness
        section_width = self.section_width
        section_depth = self.section_depth
        side_width = self.side_width
        side_height = self.side_height
        side_depth = self.side_depth
        inverted = self.inverted
        use_side_patterns = self.use_side_patterns

        # draw a line of section width
        section_y = section_thickness
        if inverted:
            section_y *= -1.0
            side_height *= -1.0

        # main section pattern
        section_pattern = FibsemLineSettings(start_x=point.x - section_width / 2, 
                                             end_x=point.x + section_width / 2, 
                                             start_y=point.y + section_y, 
                                             end_y=point.y + section_y, 
                                             depth=section_depth)
        
        self.shapes = [section_pattern]

        if use_side_patterns:
            # side cleaning patterns
            left_side_pattern = FibsemLineSettings(
                start_x=point.x - section_width / 2 - side_width / 2,
                end_x=point.x - section_width / 2 + side_width / 2,
                start_y=point.y + section_y,
                end_y=point.y + section_y,
                depth=side_depth,
            )
            right_side_pattern = FibsemLineSettings(
                start_x=point.x + section_width / 2 - side_width / 2,
                end_x=point.x + section_width / 2 + side_width / 2,
                start_y=point.y + section_y,
                end_y=point.y + section_y,
                depth=side_depth,
            )

            # side vertical patterns
            left_side_pattern_vertical = FibsemLineSettings(
                start_x=point.x - section_width / 2,
                end_x=point.x - section_width / 2,
                start_y=point.y + section_y,
                end_y=point.y + section_y + side_height,
                depth=side_depth,
            )

            right_side_pattern_vertical = FibsemLineSettings(
                start_x=point.x + section_width / 2,
                end_x=point.x + section_width / 2,
                start_y=point.y + section_y,
                end_y=point.y + section_y + side_height,
                depth=side_depth,
            )

            self.shapes.extend([left_side_pattern, right_side_pattern, 
                            left_side_pattern_vertical, 
                            right_side_pattern_vertical])
            
        return self.shapes

    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "section_thickness": self.section_thickness,
            "section_width": self.section_width,
            "section_depth": self.section_depth,
            "side_width": self.side_width,
            "side_height": self.side_height,
            "side_depth": self.side_depth,
            "inverted": self.inverted,
            "use_side_patterns": self.use_side_patterns
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "SerialSectionPattern":
        return cls(
            section_thickness=ddict["section_thickness"],
            section_width=ddict["section_width"],
            section_depth=ddict["section_depth"],
            side_width=ddict["side_width"],
            side_height=ddict.get("side_height", 0),
            side_depth=ddict.get("side_depth", 0),
            inverted=ddict.get("inverted", False),
            use_side_patterns=ddict.get("use_side_patterns", True),
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )


@dataclass
class FiducialPattern(BasePattern):
    width: float
    height: float
    depth: float
    rotation: float = 0
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle
    point: Point = Point()
    shapes: List[FibsemPatternSettings] = None
    name: str = "Fiducial"

    def define(self) -> List[FibsemRectangleSettings]:
        """Draw a fiducial milling pattern (cross shape)"""

        width = self.width
        height = self.height
        depth = self.depth
        rotation = self.rotation * constants.DEGREES_TO_RADIANS
        cross_section = self.cross_section


        left_pattern = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            rotation=rotation,
        )
        right_pattern = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=self.point.x,
            centre_y=self.point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section,
            rotation=rotation + np.deg2rad(90),
        )

        self.shapes = [left_pattern, right_pattern]
        return self.shapes

    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "rotation": self.rotation,
            "cross_section": self.cross_section.name
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "FiducialPattern":
        return cls(
            width=ddict["width"],
            height=ddict["height"],
            depth=ddict["depth"],
            rotation=ddict.get("rotation", 0),
            cross_section=CrossSectionPattern[ddict.get("cross_section", "Rectangle")],
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )


@dataclass
class UndercutPattern(BasePattern):
    width: float
    height: float
    depth: float
    trench_width: float
    rhs_height: float
    h_offset: float
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle
    name: str = "Undercut"
    point: Point = Point()
    shapes: List[FibsemPatternSettings] = None

    def define(self) -> List[FibsemRectangleSettings]:
        
        point = self.point
        jcut_rhs_height = self.rhs_height
        jcut_lamella_height = self.height
        jcut_width = self.width
        jcut_trench_thickness = self.trench_width
        jcut_depth = self.depth
        jcut_h_offset = self.h_offset
        cross_section = self.cross_section

        # top_jcut
        jcut_top_centre_x = point.x + jcut_width / 2 - jcut_h_offset
        jcut_top_centre_y = point.y + jcut_lamella_height
        jcut_top_width = jcut_width
        jcut_top_height = jcut_trench_thickness
        jcut_top_depth = jcut_depth

        top_pattern = FibsemRectangleSettings(
            width=jcut_top_width,
            height=jcut_top_height,
            depth=jcut_top_depth,
            centre_x=point.x,
            centre_y=point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section
        )
        # rhs jcut
        jcut_rhs_centre_x = point.x + (jcut_width / 2) - jcut_trench_thickness / 2
        jcut_rhs_centre_y = point.y - (jcut_rhs_height / 2) + jcut_trench_thickness / 2
        jcut_rhs_width = jcut_trench_thickness
        jcut_rhs_height = jcut_rhs_height
        jcut_rhs_depth = jcut_depth

        rhs_pattern = FibsemRectangleSettings(
            width=jcut_rhs_width,
            height=jcut_rhs_height,
            depth=jcut_rhs_depth,
            centre_x=jcut_rhs_centre_x,
            centre_y=jcut_rhs_centre_y,
            scan_direction="TopToBottom",
            cross_section = cross_section
        )

        self.shapes = [top_pattern, rhs_pattern]
        return self.shapes

    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "trench_width": self.trench_width,
            "rhs_height": self.rhs_height,
            "h_offset": self.h_offset,
            "cross_section": self.cross_section.name
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "UndercutPattern":
        return cls(
            width=ddict["width"],
            height=ddict["height"],
            depth=ddict["depth"],
            trench_width=ddict["trench_width"],
            rhs_height=ddict["rhs_height"],
            h_offset=ddict["h_offset"],
            cross_section=CrossSectionPattern[ddict.get("cross_section", "Rectangle")],
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )

@dataclass
class MicroExpansionPattern(BasePattern):
    width: float
    height: float
    depth: float
    distance: float
    name: str = "MicroExpansion"
    point: Point = Point()
    shapes: List[FibsemPatternSettings] = None

    # ref: https://www.nature.com/articles/s41467-022-29501-3
    def define(self) -> List[FibsemRectangleSettings]:
        """Draw the microexpansion joints for stress relief of lamella"""

        point = self.point
        width = self.width
        height = self.height
        depth = self.depth
        distance = self.distance

        left_pattern_settings = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=point.x -  distance,
            centre_y=point.y,
            scan_direction="TopToBottom",
        )

        right_pattern_settings = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=point.x + distance,
            centre_y=point.y,
            scan_direction="TopToBottom",
        )

        self.shapes = [left_pattern_settings, right_pattern_settings]
        return self.shapes
    
    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "distance": self.distance
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "MicroExpansionPattern":
        return cls(
            width=ddict["width"],
            height=ddict["height"],
            depth=ddict["depth"],
            distance=ddict["distance"],
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )

@dataclass
class ArrayPattern(BasePattern):
    width: float
    height: float
    depth: float
    n_columns: int
    n_rows: int
    pitch_vertical: float
    pitch_horizontal: float
    passes: int = 0
    rotation: float = 0
    scan_direction: str = "TopToBottom"
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle
    name: str = "ArrayPattern"
    point: Point = Point()
    shapes: List[FibsemPatternSettings] = None
    # ref: spotweld terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14
    # ref: weld cross-section/ passes: https://www.nature.com/articles/s41592-023-02113-5

    def define(self) -> List[FibsemRectangleSettings]:
        
        point = self.point
        width = self.width
        height = self.height
        depth = self.depth
        n_columns = int(self.n_columns)
        n_rows = int(self.n_rows)
        pitch_vertical = self.pitch_vertical
        pitch_horizontal = self.pitch_horizontal
        rotation = self.rotation
        passes = self.passes
        scan_direction = self.scan_direction
        passes = int(passes)
        cross_section = self.cross_section

        # create a 2D array of points
        points: List[Point] = []
        for i in range(n_columns):
            for j in range(n_rows):
                points.append(
                    Point(
                        point.x + (i - (n_columns - 1) / 2) * pitch_horizontal,
                        point.y + (j - (n_rows - 1) / 2) * pitch_vertical,
                    )
                )
        # create patterns
        self.shapes = []
        for pt in points:
            pattern_settings = FibsemRectangleSettings(
                width=width,
                height=height,
                depth=depth,
                centre_x=pt.x,
                centre_y=pt.y,  
                scan_direction=scan_direction,
                rotation=rotation,
                passes=passes,
                cross_section=cross_section,
            )
            self.shapes.append(pattern_settings)

        return self.shapes

    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "n_columns": self.n_columns,
            "n_rows": self.n_rows,
            "pitch_vertical": self.pitch_vertical,
            "pitch_horizontal": self.pitch_horizontal,
            "passes": self.passes,
            "rotation": self.rotation,
            "scan_direction": self.scan_direction,
            "cross_section": self.cross_section.name
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "ArrayPattern":
        return cls(
            width=ddict["width"],
            height=ddict["height"],
            depth=ddict["depth"],
            n_columns=ddict["n_columns"],
            n_rows=ddict["n_rows"],
            pitch_vertical=ddict["pitch_vertical"],
            pitch_horizontal=ddict["pitch_horizontal"],
            passes=ddict.get("passes", 0),
            rotation=ddict.get("rotation", 0),
            scan_direction=ddict.get("scan_direction", "TopToBottom"),
            cross_section=CrossSectionPattern[ddict.get("cross_section", "Rectangle")],
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )

@dataclass
class WaffleNotchPattern(BasePattern):
    vheight: float
    vwidth: float
    hheight: float
    hwidth: float
    depth: float
    distance: float
    inverted: bool = False
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle
    name: str = "WaffleNotch"
    point: Point = Point()
    shapes: List[FibsemPatternSettings] = None
    # ref: https://www.nature.com/articles/s41467-022-29501-3

    def define(self) -> List[FibsemRectangleSettings]:

        point = self.point
        vwidth = self.vwidth
        vheight = self.vheight
        hwidth = self.hwidth
        hheight = self.hheight
        depth = self.depth
        distance = self.distance
        cross_section = self.cross_section
        inverted = -1 if self.inverted else 1

        # five patterns
        top_vertical_pattern = FibsemRectangleSettings(
            width=vwidth,
            height=vheight,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y - distance / 2 - vheight / 2 + hheight / 2,
            scan_direction="TopToBottom",
            cross_section=cross_section
        )

        bottom_vertical_pattern = FibsemRectangleSettings(
            width=vwidth,
            height=vheight,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y + distance / 2 + vheight / 2 - hheight / 2,
            scan_direction="BottomToTop",
            cross_section=cross_section
        )

        top_horizontal_pattern = FibsemRectangleSettings(
            width=hwidth,
            height=hheight,
            depth=depth,
            centre_x=point.x + (hwidth / 2 + vwidth / 2) * inverted,
            centre_y=point.y - distance / 2,
            scan_direction="TopToBottom",
            cross_section=cross_section
        )

        bottom_horizontal_pattern = FibsemRectangleSettings(
            width=hwidth,
            height=hheight,
            depth=depth,
            centre_x=point.x + (hwidth / 2 + vwidth / 2) * inverted,
            centre_y=point.y + distance / 2,
            scan_direction="BottomToTop",
            cross_section=cross_section
        )

        centre_vertical_pattern = FibsemRectangleSettings(
            width=vwidth,
            height=distance + hheight,
            depth=depth,
            centre_x=point.x + (hwidth + vwidth) * inverted,
            centre_y=point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section
        )

        self.shapes = [
            top_vertical_pattern,
            bottom_vertical_pattern,
            top_horizontal_pattern,
            bottom_horizontal_pattern,
            centre_vertical_pattern,
        ]

        return self.shapes

    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "vheight": self.vheight,
            "vwidth": self.vwidth,
            "hheight": self.hheight,
            "hwidth": self.hwidth,
            "depth": self.depth,
            "distance": self.distance,
            "inverted": self.inverted,
            "cross_section": self.cross_section.name
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "WaffleNotchPattern":
        return cls(
            vheight=ddict["vheight"],
            vwidth=ddict["vwidth"],
            hheight=ddict["hheight"],
            hwidth=ddict["hwidth"],
            depth=ddict["depth"],
            distance=ddict["distance"],
            inverted=ddict.get("inverted", False),
            cross_section=CrossSectionPattern[ddict.get("cross_section", "Rectangle")],
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )

@dataclass
class CloverPattern(BasePattern):
    radius: float
    depth: float
    point: Point = Point()
    shapes: List[FibsemPatternSettings] = None
    name: str = "Clover"


    def define(self) -> List[FibsemPatternSettings]:
        
        point = self.point
        radius = self.radius
        depth = self.depth

        # three leaf clover pattern

        top_pattern = FibsemCircleSettings(
            radius=radius,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y + radius,
        )

        right_pattern = FibsemCircleSettings(
            radius=radius,
            depth=depth,
            centre_x=point.x + radius,
            centre_y=point.y,
        )

        left_pattern = FibsemCircleSettings(
            radius=radius,
            depth=depth,
            centre_x=point.x - radius,
            centre_y=point.y,
        )

        stem_pattern = FibsemRectangleSettings(
            width=radius / 4,
            height=radius * 2,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y - radius,
            scan_direction="TopToBottom",
        )

        self.shapes = [top_pattern, right_pattern, left_pattern, stem_pattern]
        return self.shapes

    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "radius": self.radius,
            "depth": self.depth
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "CloverPattern":
        return cls(
            radius=ddict["radius"],
            depth=ddict["depth"],
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )


@dataclass
class TriForcePattern(BasePattern):
    width: float
    height: float
    depth: float
    point: Point = Point()
    shapes: List[FibsemPatternSettings] = None
    name: str = "TriForce"

    def define(self) -> List[FibsemPatternSettings]:
        
        point = self.point
        height = self.height
        width = self.width
        depth = self.depth
        angle = 30

        self.shapes = []

        # centre of each triangle
        points = [
            Point(point.x, point.y + height),
            Point(point.x - height / 2, point.y),
            Point(point.x + height / 2, point.y),
        ]

        for point in points:

            triangle_shapes =  create_triangle_patterns(width=width, 
                                                        height=height, 
                                                        depth=depth, 
                                                        point=point, 
                                                        angle=angle)
            self.shapes.extend(triangle_shapes)
            
        return self.shapes

    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "width": self.width,
            "height": self.height,
            "depth": self.depth
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "TriForcePattern":
        return cls(
            width=ddict["width"],
            height=ddict["height"],
            depth=ddict["depth"],
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )

@dataclass
class TrapezoidPattern(BasePattern):
    inner_width: float
    outer_width: float
    trench_height: float
    depth: float
    distance: float
    n_rectangles: int
    overlap: float
    name: str = "Trapezoid"
    point: Point = Point()
    shapes: List[FibsemPatternSettings] = None

    def define(self) -> List[FibsemPatternSettings]:
        
        outer_width = self.outer_width
        inner_width = self.inner_width
        trench_height = self.trench_height
        depth = self.depth
        distance = self.distance
        n_rectangles = int(self.n_rectangles)
        overlap = self.overlap
        point = self.point

        self.shapes = []
        width_increments = (outer_width - inner_width) / (n_rectangles - 1)
        height = trench_height / n_rectangles * (1 + overlap)
        # bottom half
        for i in range(n_rectangles):
            width = outer_width - i * width_increments
            y = point.y + (i * dict["trench_height"] * (1-overlap)) - distance - trench_height
            centre = Point(point.x, y)
            pattern = FibsemRectangleSettings(
                width=width,
                height=height,
                depth=depth,
                centre_x=centre.x,
                centre_y=centre.y,
                scan_direction="BottomToTop",
            )
            self.shapes.append(deepcopy(pattern))
        # top half
        for i in range(n_rectangles):
            width = outer_width - i * width_increments
            y = point.y - (i * dict["trench_height"] * (1-overlap)) + distance + trench_height
            centre = Point(point.x, y)
            pattern = FibsemRectangleSettings(
                width=width,
                height=height,
                depth=depth,
                centre_x=centre.x,
                centre_y=centre.y,
                scan_direction="TopToBottom",
            )
            self.shapes.append(deepcopy(pattern))
        return self.shapes
    
    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict(),
            "inner_width": self.inner_width,
            "outer_width": self.outer_width,
            "trench_height": self.trench_height,
            "depth": self.depth,
            "distance": self.distance,
            "n_rectangles": self.n_rectangles,
            "overlap": self.overlap
        }
    
    @classmethod
    def from_dict(cls, ddict: dict) -> "TrapezoidPattern":

        return cls(
            inner_width=ddict["inner_width"],
            outer_width=ddict["outer_width"],
            trench_height=ddict["trench_height"],
            depth=ddict["depth"],
            distance=ddict["distance"],
            n_rectangles=ddict["n_rectangles"],
            overlap=ddict["overlap"],
            point=Point.from_dict(ddict.get("point", DEFAULT_POINT_DDICT))
        )


def create_triangle_patterns(
    width: float, height: float, depth: float, angle: float = 30, point: Point = Point()
) -> List[FibsemRectangleSettings]:
    h_offset = height / 2 * np.sin(np.deg2rad(angle))

    left_pattern = FibsemRectangleSettings(
        width=width,
        height=height,
        depth=depth,
        rotation=np.deg2rad(-angle),
        centre_x=point.x - h_offset,
        centre_y=point.y,
        scan_direction="LeftToRight",
    )

    right_pattern = FibsemRectangleSettings(
        width=width,
        height=height,
        depth=depth,
        rotation=np.deg2rad(angle),
        centre_x=point.x + h_offset,
        centre_y=point.y,
        scan_direction="RightToLeft",
    )

    bottom_pattern = FibsemRectangleSettings(
        width=width,
        height=height,
        depth=depth,
        rotation=np.deg2rad(90),
        centre_x=point.x,
        centre_y=point.y - height / 2,
        scan_direction="BottomToTop",
    )

    return [left_pattern, right_pattern, bottom_pattern]


MILLING_PATTERNS: Dict[str, BasePattern] = {
    RectanglePattern.name.lower(): RectanglePattern,
    LinePattern.name.lower(): LinePattern,
    CirclePattern.name.lower(): CirclePattern,
    TrenchPattern.name.lower(): TrenchPattern,
    HorseshoePattern.name.lower(): HorseshoePattern,
    HorseshoePatternVertical.name.lower(): HorseshoePatternVertical,
    SerialSectionPattern.name.lower(): SerialSectionPattern,
    UndercutPattern.name.lower(): UndercutPattern,
    FiducialPattern.name.lower(): FiducialPattern,
    ArrayPattern.name.lower(): ArrayPattern,
    MicroExpansionPattern.name.lower(): MicroExpansionPattern,
    WaffleNotchPattern.name.lower(): WaffleNotchPattern,
    CloverPattern.name.lower(): CloverPattern,
    TriForcePattern.name.lower(): TriForcePattern,
    # TrapezoidPattern.name.lower(): TrapezoidPattern,
    # BitmapPattern.name.lower(): BitmapPattern,
}
MILLING_PATTERN_NAMES = [p.name for p in MILLING_PATTERNS.values()]
DEFAULT_MILLING_PATTERN = RectanglePattern.name

# legacy mapping
PROTOCOL_MILL_MAP = {
    "cut": RectanglePattern,
    "fiducial": FiducialPattern,
    "flatten": RectanglePattern,
    "undercut": UndercutPattern,
    "horseshoe": HorseshoePattern,
    "lamella": TrenchPattern,
    "polish_lamella": TrenchPattern,
    "thin_lamella": TrenchPattern,
    "sever": RectanglePattern,
    "sharpen": RectanglePattern,
    "needle": RectanglePattern,
    "copper_release": HorseshoePattern,
    "serial_trench": HorseshoePattern,
    "serial_undercut": RectanglePattern,
    "serial_sever": RectanglePattern,
    "lamella_sever": RectanglePattern,
    "lamella_polish": TrenchPattern,
    "trench": TrenchPattern,
    "notch": WaffleNotchPattern,
    "microexpansion": MicroExpansionPattern,
    "clover": CloverPattern,
    "autolamella": TrenchPattern,
    "MillUndercut": RectanglePattern,
    "rectangle": RectanglePattern,
    "MillRough": TrenchPattern,
    "MillRegularCut": TrenchPattern,
    "MillPolishing": TrenchPattern,
    "mill_rough": TrenchPattern,
    "mill_polishing": TrenchPattern,
}

def get_pattern(name: str, config: dict) -> BasePattern:
    cls_pattern = MILLING_PATTERNS.get(name.lower())
    pattern = cls_pattern.from_dict(config)
    return pattern