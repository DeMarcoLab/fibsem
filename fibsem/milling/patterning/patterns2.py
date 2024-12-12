import json
from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import yaml

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


def check_keys(protocol: dict, required_keys: List[str]) -> bool:
    return all([k in protocol.keys() for k in required_keys])

# TODO: define the configuration for each key,
# e.g. 
# "width": {"type": "float", "min": 0, "max": 1000, "default": 100, "description": "Width of the rectangle"}
# "cross_section": {"type": "str", "options": [cs.name for cs in CrossSectionPattern], "default": "Rectangle", "description": "Cross section of the milling pattern"}

    
REQUIRED_KEYS = {
    "Rectangle": ("width", "height", "depth", "rotation","passes", "scan_direction", "cross_section", "time"),
    "Line": ("start_x", "end_x", "start_y", "end_y", "depth"),
    "Circle": ("radius", "depth"),
    "Trench": (
        "lamella_width",
        "lamella_height",
        "trench_height",
        "size_ratio",
        "offset",
        "depth",
        "cross_section",
        "time", 
    ),
    "Horseshoe": (
        "lamella_width",
        "lamella_height",
        "trench_height",
        "size_ratio",
        "offset",
        "side_offset",
        "side_width",
        "depth",
        "scan_direction",
        "cross_section",
    ),
    "HorseshoeVertical": (
        "width",
        "height",
        "side_trench_width",
        "top_trench_height",
        "depth",
        "scan_direction",
        "inverted",
        "cross_section", 
    ),
    "SerialSection": (
        "section_thickness",
        "section_width",
        "section_depth",
        "side_width",
        "side_height",
        "side_depth",
        "inverted",
        "use_side_patterns",
    ),
    "RectangleOffset": ("width", "height", "depth", "scan_direction", "cross_section", "offset", "inverted"),
    "Fiducial": ("height", "width", "depth", "rotation", "cross_section"),
    "Undercut": (
        "height",
        "width",
        "depth",
        "trench_width",
        "rhs_height",
        "h_offset",
        "cross_section",
    ),
    "MicroExpansion": (
        "height",
        "width",
        "depth",
        "distance",
    ),
    "ArrayPattern": ("height", "width", "depth", "n_columns", "n_rows", 
                    "pitch_vertical", "pitch_horizontal", 
                    "passes", "scan_direction", "cross_section"),
    "WaffleNotch": (
        "vheight",
        "vwidth",
        "hheight",
        "hwidth",
        "depth",
        "distance",
        "inverted",
        "cross_section",
    ),
    "Clover": ("radius", "depth"),
    "TriForce": ("height", "width", "depth"),
    "Trapezoid": ("inner_width", "outer_width", "trench_height", "depth", "distance", "n_rectangles", "overlap"),
}

DEFAULT_POINT_DDICT = {"x": 0.0, "y": 0.0}

####### Combo Patterns

@dataclass
class BasePattern(ABC):
    name: str = "BasePattern"
    shapes: List[FibsemPatternSettings] = None
    point: Point = Point()

    @abstractmethod
    def define(self) -> List[FibsemPatternSettings]:
        pass

    def to_dict(self):
        return {
            "name": self.name,
            "point": self.point.to_dict()
        }

    @abstractmethod
    @classmethod
    def from_dict(cls, ddict: dict) -> "BasePattern":
        pass



@dataclass
class BitmapPattern(BasePattern):
    width: float
    height: float
    depth: float
    rotation: float = 0
    path: str = ""
    shapes: List[FibsemPatternSettings] = None
    point: Point = Point()
    name: str = "BitmapPattern"

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
            point=Point.from_dict(ddict["point"])
        )


@dataclass
class RectanglePattern(BasePattern):
    width: float
    height: float
    depth: float
    rotation: float = 0
    time: float = 0
    passes: int = 0
    offset: float = 0
    invert_offset: bool = False
    scan_direction: str = "TopToBottom"
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle
    point: Point = Point()
    shapes: List[FibsemPatternSettings] = None
    name: str = "Rectangle"

    def define(self) -> List[FibsemRectangleSettings]:

        # allow for offsetting the rectangle from the point
        offset = self.offset + self.height / 2
        if self.invert_offset:
            offset = -offset
        center_y = self.point.y + offset

        shape = FibsemRectangleSettings(
            width=self.width,
            height=self.height,
            depth=self.depth,
            centre_x=self.point.x,
            centre_y=center_y,
            rotation=self.rotation * constants.DEGREES_TO_RADIANS,
            passes=self.passes,
            time=self.time,
            scan_direction=self.scan_direction,
            cross_section=self.cross_section,
        )

        self.shapes = [shape]
        return self.shapes


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


@dataclass
class HorseshoePattern(BasePattern):
    width: float
    spacing: float
    depth: float
    upper_trench_height: float
    lower_trench_height: float
    side_width: float
    side_offset: float
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
        side_offset = self.side_offset

        # calculate the centre of the upper and lower trench
        centre_upper_y = point.y + (spacing / 2 + upper_trench_height / 2)
        centre_lower_y = point.y - (spacing / 2 + lower_trench_height / 2)

        # calculate the centre of the side trench
        side_x = point.x - side_offset + (width / 2 - side_width / 2)

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
            height=width,
            depth=depth,
            centre_x=side_x,
            centre_y=point.y,
            scan_direction="TopToBottom",
            cross_section=cross_section
        )

        self.shapes = [lower_pattern, upper_pattern, side_pattern]
        return self.shapes

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
        upper_trench_height = self.upper_trench_height
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

# TODO: deprecate this pattern, add offset to RectanglePattern
# @dataclass
# class RectangleOffsetPattern(BasePattern):
#     name: str = "RectangleOffset"
#     required_keys: Tuple[str] = REQUIRED_KEYS["RectangleOffset"]
#     patterns = None
#     protocol = None
#     point = None

#     def define(
#         self, protocol: dict, point: Point = Point()
#     ) -> List[FibsemRectangleSettings]:
#         check_keys(protocol, self.required_keys)

#         width = protocol["width"]
#         height = protocol["height"]
#         depth = protocol["depth"]
#         offset = protocol["offset"]
#         scan_direction = protocol.get("scan_direction", "TopToBottom")
#         cross_section = CrossSectionPattern[protocol.get("cross_section", "Rectangle")]
#         inverted = protocol.get("inverted", False)

#         offset = offset + height / 2
#         if inverted:
#             offset = -offset
            
#         center_y = point.y + offset

#         pattern = FibsemRectangleSettings(
#             width=width,
#             height=height,
#             depth=depth,
#             centre_x=point.x,
#             centre_y=center_y,
#             scan_direction=scan_direction,
#             cross_section = cross_section,
#         )

#         self.patterns = [pattern]
#         self.protocol = protocol
#         self.point = point
#         return self.patterns

@dataclass
class FiducialPattern(BasePattern):
    name: str = "Fiducial"
    required_keys: Tuple[str] = REQUIRED_KEYS["Fiducial"]
    patterns = None
    protocol = None
    point = None

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemRectangleSettings]:
        import numpy as np

        # ?
        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["pattern"] = "Rectangle"
        protocol["cleaning_cross_section"] = protocol.get(
            "cleaning_cross_section", False
        )
        protocol["cross_section"] = protocol.get("cross_section", "Rectangle")
        protocol["scan_direction"] = protocol.get("scan_direction", "TopToBottom")

        left_pattern = FibsemRectangleSettings.from_dict(protocol)
        left_pattern.rotation = protocol["rotation"] * constants.DEGREES_TO_RADIANS
        right_pattern = FibsemRectangleSettings.from_dict(protocol)
        right_pattern.rotation = left_pattern.rotation + np.deg2rad(90)

        self.patterns = [left_pattern, right_pattern]
        self.protocol = protocol
        self.point = point
        return self.patterns


@dataclass
class UndercutPattern(BasePattern):
    name: str = "Undercut"
    required_keys: Tuple[str] = REQUIRED_KEYS["Undercut"]
    patterns = None
    protocol = None
    point = None

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemRectangleSettings]:
        check_keys(protocol, self.required_keys)

        jcut_rhs_height = protocol["rhs_height"]
        jcut_lamella_height = protocol["height"]
        jcut_width = protocol["width"]
        jcut_trench_thickness = protocol["trench_width"]
        jcut_depth = protocol["depth"]
        jcut_h_offset = protocol["h_offset"]
        cross_section = CrossSectionPattern[protocol.get("cross_section", "Rectangle")]

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
            cross_section = CrossSectionPattern[protocol.get("cross_section", "Rectangle")]
        )



        self.patterns = [top_pattern, rhs_pattern]
        self.protocol = protocol
        self.point = point
        return self.patterns


@dataclass
class MicroExpansionPattern(BasePattern):
    name: str = "MicroExpansion"
    required_keys: Tuple[str] = REQUIRED_KEYS["MicroExpansion"]
    patterns = None
    protocol = None
    point = None

    # ref: https://www.nature.com/articles/s41467-022-29501-3
    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemRectangleSettings]:
        """
        Draw the microexpansion joints for stress relief of lamella.

        Args:
            microscope (FibsemMicroscope): OpenFIBSEM microscope instance
            protocol (dict): Contains a dictionary of the necessary values for drawing the joints.
            protocol (dict): Lamella protocol

        Returns:
            patterns: List[FibsemPatternSettings]
        """
        check_keys(protocol, self.required_keys)

        width = protocol["width"]
        height = protocol["height"]
        depth = protocol["depth"]  # lamella milling depth

        left_pattern_settings = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=point.x -  protocol["distance"] ,
            centre_y=point.y,
            scan_direction="TopToBottom",
        )

        right_pattern_settings = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=point.x + protocol["distance"],
            centre_y=point.y,
            scan_direction="TopToBottom",
        )

        self.patterns = [left_pattern_settings, right_pattern_settings]
        self.protocol = protocol
        self.point = point
        return self.patterns

@dataclass
class ArrayPattern(BasePattern):
    name: str = "ArrayPattern"
    required_keys: Tuple[str] = REQUIRED_KEYS["ArrayPattern"]
    patterns = None
    # ref: spotweld terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14
    # ref: weld cross-section/ passes: https://www.nature.com/articles/s41592-023-02113-5
    protocol = None
    point = None

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemRectangleSettings]:
        check_keys(protocol, self.required_keys)

        width = protocol["width"]
        height = protocol["height"]
        depth = protocol["depth"]
        n_columns = int(protocol["n_columns"])
        n_rows = int(protocol["n_rows"])
        pitch_vertical = protocol["pitch_vertical"]
        pitch_horizontal = protocol["pitch_horizontal"]
        rotation = protocol.get("rotation", 0)
        passes = protocol.get("passes", 1)
        scan_direction = protocol.get("scan_direction", "TopToBottom")
        passes = int(passes) if passes is not None else None
        cross_section = CrossSectionPattern[protocol.get("cross_section", "Rectangle")]

        # create a 2D array of points
        points = []
        for i in range(n_columns):
            for j in range(n_rows):
                points.append(
                    Point(
                        point.x + (i - (n_columns - 1) / 2) * pitch_horizontal,
                        point.y + (j - (n_rows - 1) / 2) * pitch_vertical,
                    )
                )
        # create patterns
        patterns = []
        for point in points:
            pattern_settings = FibsemRectangleSettings(
                width=width,
                height=height,
                depth=depth,
                centre_x=point.x,
                centre_y=point.y,  
                scan_direction=scan_direction,

                rotation=rotation,
                passes=passes,
                cross_section=cross_section,
            )
            patterns.append(pattern_settings)

        self.patterns = patterns
        self.protocol = protocol
        self.point = point
        return self.patterns




@dataclass
class WaffleNotchPattern(BasePattern):
    name: str = "WaffleNotch"
    required_keys: Tuple[str] = REQUIRED_KEYS["WaffleNotch"]
    patterns = None
    protocol = None
    point = None
    # ref: https://www.nature.com/articles/s41467-022-29501-3

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemRectangleSettings]:
        check_keys(protocol, self.required_keys)

        vwidth = protocol["vwidth"]
        vheight = protocol["vheight"]
        hwidth = protocol["hwidth"]
        hheight = protocol["hheight"]
        depth = protocol["depth"]
        distance = protocol["distance"]
        cross_section = CrossSectionPattern[protocol.get("cross_section", "Rectangle")]
        inverted = -1  if protocol.get("inverted", False) else 1

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

        self.patterns = [
            top_vertical_pattern,
            bottom_vertical_pattern,
            top_horizontal_pattern,
            bottom_horizontal_pattern,
            centre_vertical_pattern,
        ]
        self.protocol = protocol
        self.point = point
        return self.patterns


@dataclass
class CloverPattern(BasePattern):
    name: str = "Clover"
    required_keys: Tuple[str] = REQUIRED_KEYS["Clover"]
    patterns = None
    protocol = None
    point = None

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemPatternSettings]:
        check_keys(protocol, self.required_keys)

        radius = protocol["radius"]
        depth = protocol["depth"]

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

        self.patterns = [top_pattern, right_pattern, left_pattern, stem_pattern]
        self.protocol = protocol
        self.point = point
        return self.patterns


@dataclass
class TriForcePattern(BasePattern):
    name: str = "TriForce"
    required_keys: Tuple[str] = REQUIRED_KEYS["TriForce"]
    patterns = None
    protocol = None
    point = None

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemPatternSettings]:
        check_keys(protocol, self.required_keys)

        height = protocol["height"]
        width = protocol["width"]
        depth = protocol["depth"]

        # triforce pattern
        angle = 30

        self.patterns = []

        # centre of each triangle
        points = [
            Point(point.x, point.y + height),
            Point(point.x - height / 2, point.y),
            Point(point.x + height / 2, point.y),
        ]

        for point in points:

            left_pattern, right_pattern, bottom_pattern = create_triangle_patterns(width=width, height=height, depth=depth, point=point, angle=angle)

            self.patterns.append(left_pattern)
            self.patterns.append(right_pattern)
            self.patterns.append(bottom_pattern)
        self.protocol = protocol
        self.point = point
        return self.patterns


@dataclass
class TrapezoidPattern(BasePattern):
    name: str = "Trapezoid"
    required_keys: Tuple[str] = REQUIRED_KEYS["Trapezoid"]
    patterns = None
    protocol = None
    point = None

    def define(self, protocol: dict, point: Point = Point()) -> List[FibsemPatternSettings]:
        check_keys(protocol, self.required_keys)
        self.patterns = []
        width_increments = (protocol["outer_width"] - protocol["inner_width"]) / (protocol["n_rectangles"]-1)
        dict = {}
        dict["trench_height"] = protocol["trench_height"] / protocol["n_rectangles"] * (1+protocol["overlap"])
        dict["depth"] = protocol["depth"]
        # bottom half
        for i in range(int(protocol["n_rectangles"])):
            dict["width"] = protocol["outer_width"] - i * width_increments
            pattern = RectanglePattern()
            y = point.y + (i * dict["trench_height"] * (1-protocol["overlap"])) - protocol["distance"] - protocol["trench_height"]
            centre = Point(point.x, y)
            pattern = FibsemRectangleSettings(
                width=dict["width"],
                height=dict["trench_height"],
                depth=dict["depth"],
                centre_x=centre.x,
                centre_y=centre.y,
                scan_direction="BottomToTop",
            )
            self.patterns.append(deepcopy(pattern))
        # top half
        for i in range(int(protocol["n_rectangles"])):
            dict["width"] = protocol["outer_width"] - i * width_increments
            pattern = RectanglePattern()
            y = point.y - (i * dict["trench_height"] * (1-protocol["overlap"])) + protocol["distance"] + protocol["trench_height"]
            centre = Point(point.x, y)
            pattern = FibsemRectangleSettings(
                width=dict["width"],
                height=dict["trench_height"],
                depth=dict["depth"],
                centre_x=centre.x,
                centre_y=centre.y,
                scan_direction="TopToBottom",
            )
            self.patterns.append(deepcopy(pattern))
        self.protocol = protocol
        self.point = point
        return self.patterns


__PATTERNS__ = [
    RectanglePattern,
    LinePattern,
    CirclePattern,
    TrenchPattern,
    HorseshoePattern,
    HorseshoePatternVertical,
    SerialSectionPattern,
    UndercutPattern,
    FiducialPattern,
    ArrayPattern,
    MicroExpansionPattern,
    WaffleNotchPattern,
    CloverPattern,
    TriForcePattern,
    BitmapPattern,
    TrapezoidPattern,
]


def get_pattern(name: str) -> BasePattern:
    for pattern in __PATTERNS__:
        if pattern.name.lower() == name.lower():
            return pattern()

    raise ValueError(f"Pattern {name} not found.")


def get_pattern_names() -> List[str]:
    return [pattern.name for pattern in __PATTERNS__]


def get_pattern_required_keys(name: str) -> Tuple[str]:
    pattern = get_pattern(name)
    return pattern.required_keys


def get_pattern_required_keys_dict() -> Dict[str, List[str]]:
    return {pattern.name: pattern.required_keys for pattern in __PATTERNS__}


def get_pattern_required_keys_json() -> str:
    return json.dumps(get_pattern_required_keys_dict())


def get_pattern_required_keys_yaml() -> str:
    return yaml.dump(get_pattern_required_keys_dict())


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
    "MillRoughCut": TrenchPattern,
    "MillRegularCut": TrenchPattern,
    "MillPolishingCut": TrenchPattern,
    "mill_rough": TrenchPattern,
    "mill_polishing": TrenchPattern,

}


def _get_pattern(key: str, protocol: dict, point: Point = Point()) -> BasePattern:
    if "type" in protocol:
        pattern = get_pattern(protocol["type"])
    else:
        pattern = PROTOCOL_MILL_MAP[key]()
    pattern.define(protocol, point=point)
    return pattern

