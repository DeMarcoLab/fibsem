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
    FibsemMillingSettings,
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


####### Combo Patterns

@dataclass
class BasePattern(ABC):
    name: str = "BasePattern"
    required_keys: Tuple[str] = ()
    patterns = None
    protocol = None
    point: Point = Point()

    @abstractmethod
    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemPatternSettings]:
        pass

    def to_dict(self):
        return {
            "name": self.name,
            "required_keys": self.required_keys,
            "protocol": self.protocol,
            "point": self.point.to_dict() if self.point is not None else None,
        }

    @classmethod
    def from_dict(cls, data):
        pattern = cls(
            name=data["name"],
            required_keys=REQUIRED_KEYS[data["name"]],
        )
        pattern.protocol = data["protocol"]
        pattern.point = (
            Point.from_dict(data["point"]) if data["point"] is not None else Point()
        )
        return pattern


@dataclass
class BitmapPattern(BasePattern):
    name: str = "BitmapPattern"
    required_keys: Tuple[str] = ("width", "height", "depth", "rotation","path")
    patterns = None
    protocol = None

    def define(
            self, protocol: dict, point: Point = Point()
    ) -> List[FibsemPatternSettings]:
        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["pattern"] = "BitmapPattern"  
        protocol["rotation"] = protocol.get("rotation", 0)
        protocol["cleaning_cross_section"] = protocol.get("cleaning_cross_section", False)
        protocol["scan_direction"] = protocol.get("scan_direction", "TopToBottom")
        protocol["path"] = protocol.get("path", 'bmp_path')
        self.patterns = [FibsemBitmapSettings.from_dict(protocol)]
        self.protocol = protocol
        return self.patterns


@dataclass
class RectanglePattern(BasePattern):
    name: str = "Rectangle"
    required_keys: Tuple[str] = REQUIRED_KEYS["Rectangle"]
    patterns = None
    protocol = None
    point = None

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemRectangleSettings]:
        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["pattern"] = "Rectangle"  # redundant now
        protocol["rotation"] = protocol.get("rotation", 0) * constants.DEGREES_TO_RADIANS
        protocol["cleaning_cross_section"] = protocol.get(
            "cleaning_cross_section", False
        )
        protocol["cross_section"] = protocol.get("cross_section", "Rectangle")
        protocol["scan_direction"] = protocol.get("scan_direction", "TopToBottom")
        protocol["passes"] = protocol.get("passes", None)
        self.patterns = [FibsemRectangleSettings.from_dict(protocol)]
        self.protocol = protocol
        self.point = point
        return self.patterns


@dataclass
class LinePattern(BasePattern):
    name: str = "Line"
    required_keys: Tuple[str] = REQUIRED_KEYS["Line"]
    patterns = None
    protocol = None
    point = None

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemLineSettings]:
        protocol["pattern"] = "Line"  # redundant now
        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["cleaning_cross_section"] = protocol.get(
            "cleaning_cross_section", False
        )
        protocol["scan_direction"] = protocol.get("scan_direction", "TopToBottom")
        self.patterns = [FibsemLineSettings.from_dict(protocol)]
        self.protocol = protocol
        self.point = point
        return self.patterns


@dataclass
class CirclePattern(BasePattern):
    name: str = "Circle"
    required_keys: Tuple[str] = REQUIRED_KEYS["Circle"]
    patterns = None
    protocol = None
    point = None

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemCircleSettings]:
        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["pattern"] = "Circle"  # redundant now
        protocol["start_angle"] = 0
        protocol["end_angle"] = 360
        protocol["rotation"] = 0
        protocol["cleaning_cross_section"] = protocol.get(
            "cleaning_cross_section", False
        )
        protocol["scan_direction"] = protocol.get("scan_direction", "TopToBottom")
        self.patterns = [FibsemCircleSettings.from_dict(protocol)]
        self.protocol = protocol
        self.point = point
        return self.patterns

@dataclass
class AnnulusPattern(BasePattern):
    name: str = "Annulus"
    required_keys: Tuple[str] = ("thickness", "radius", "depth")
    patterns = None
    protocol = None
    point = None
    def define(
            self, protocol: dict, point: Point = Point()
    ) -> List[FibsemCircleSettings]:
        
        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["pattern"] = "Annulus"  # redundant now
        protocol["start_angle"] = 0
        protocol["end_angle"] = 360
        protocol["rotation"] = 0
        protocol["cleaning_cross_section"] = protocol.get("cleaning_cross_section", False)
        protocol["scan_direction"] = protocol.get("scan_direction", "TopToBottom")


        self.patterns = [FibsemCircleSettings.from_dict(protocol)]
        self.protocol = protocol
        self.point = point
        return self.patterns


@dataclass
class TrenchPattern(BasePattern):
    name: str = "Trench"
    required_keys: Tuple[str] = REQUIRED_KEYS["Trench"]
    patterns = None
    protocol = None
    point = None

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemRectangleSettings]:
        check_keys(protocol, self.required_keys)

        lamella_width = protocol["lamella_width"]
        lamella_height = protocol["lamella_height"]
        trench_height = protocol["trench_height"]
        size_ratio = protocol.get("size_ratio", 1.0)
        upper_trench_height = trench_height / max(size_ratio, 1.0)
        lower_trench_height = trench_height * min(size_ratio, 1.0)
        offset = protocol.get("offset", 0)
        depth = protocol["depth"]
        use_cleaning_cross_section = protocol.get("cleaning_cross_section", False)
        cross_section = CrossSectionPattern[protocol.get("cross_section", "Rectangle")]
        time = protocol.get("time", 0.0)

        centre_upper_y = point.y + (
            lamella_height / 2 + upper_trench_height / 2 + offset
        )
        centre_lower_y = point.y - (lamella_height / 2 + lower_trench_height / 2 + offset)

        # mill settings
        lower_pattern_settings = FibsemRectangleSettings(
            width=lamella_width,
            height=lower_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_lower_y,
            cleaning_cross_section=use_cleaning_cross_section,
            scan_direction="BottomToTop",
            cross_section = cross_section,
            time = time

        )

        upper_pattern_settings = FibsemRectangleSettings(
            width=lamella_width,
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_upper_y,
            cleaning_cross_section=use_cleaning_cross_section,
            scan_direction="TopToBottom",
            cross_section = cross_section,
            time = time
        )

        self.patterns = [lower_pattern_settings, upper_pattern_settings]
        self.protocol = protocol
        self.point = point
        return self.patterns


@dataclass
class HorseshoePattern(BasePattern):
    name: str = "Horseshoe"
    required_keys: Tuple[str] = REQUIRED_KEYS["Horseshoe"]
    patterns = None
    # ref: "horseshoe" terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14
    protocol = None
    point = None

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemRectangleSettings]:
        """Calculate the trench milling patterns"""

        check_keys(protocol, self.required_keys)

        lamella_width = protocol["lamella_width"]
        lamella_height = protocol["lamella_height"]
        trench_height = protocol["trench_height"]
        upper_trench_height = trench_height / max(protocol["size_ratio"], 1.0)
        offset = protocol["offset"]
        depth = protocol["depth"]
        cross_section = CrossSectionPattern[protocol.get("cross_section", "Rectangle")]


        centre_upper_y = point.y + (
            lamella_height / 2 + upper_trench_height / 2 + offset
        )
        centre_lower_y = point.y - (lamella_height / 2 + trench_height / 2 + offset)

        lower_pattern = FibsemRectangleSettings(
            width=lamella_width,
            height=trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_lower_y,
            cleaning_cross_section=False,
            scan_direction="BottomToTop",
            cross_section = cross_section
        )

        upper_pattern = FibsemRectangleSettings(
            width=lamella_width,
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_upper_y,
            cleaning_cross_section=False,
            scan_direction="TopToBottom",
            cross_section = cross_section
        )

        side_pattern = FibsemRectangleSettings(
            width=protocol["side_width"],
            height=lamella_height + offset,
            depth=depth,
            centre_x=point.x
            - protocol["side_offset"]
            + (lamella_width / 2 - protocol["side_width"] / 2),
            centre_y=point.y,
            cleaning_cross_section=False,
            scan_direction="TopToBottom",
            cross_section=cross_section
        )

        self.patterns = [lower_pattern, upper_pattern, side_pattern]
        self.protocol = protocol
        self.point = point
        return self.patterns



@dataclass
class HorseshoePatternVertical(BasePattern):
    name: str = "HorseshoeVertical"
    required_keys: Tuple[str] = REQUIRED_KEYS["HorseshoeVertical"]
    patterns = None
    # ref: "horseshoe" terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14
    protocol = None
    point = None

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemRectangleSettings]:
        """Calculate the horseshoe vertical milling patterns"""

        check_keys(protocol, self.required_keys)

        width = protocol["width"]
        height = protocol["height"]
        trench_width = protocol["side_trench_width"]
        upper_trench_height = protocol["top_trench_height"]
        depth = protocol["depth"]
        scan_direction = protocol.get("scan_direction", "TopToBottom")
        inverted = protocol.get("inverted", False)
        cross_section=CrossSectionPattern[protocol.get("cross_section", "Rectangle")]

        left_pattern = FibsemRectangleSettings(
            width=trench_width,
            height=height,
            depth=depth,
            centre_x=point.x - (width / 2) - (trench_width / 2),
            centre_y=point.y,
            cleaning_cross_section=False,
            scan_direction="LeftToRight",
            cross_section=cross_section
        )

        right_pattern = FibsemRectangleSettings(
            width=trench_width,
            height=height,
            depth=depth,
            centre_x=point.x + (width / 2) + (trench_width / 2),
            centre_y=point.y,
            cleaning_cross_section=False,
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
            cleaning_cross_section=False,
            scan_direction=scan_direction,
            cross_section=cross_section
        )
        
        self.patterns = [left_pattern,right_pattern, upper_pattern]
        self.protocol = protocol
        self.point = point
        return self.patterns

@dataclass
class SerialSectionPattern(BasePattern):
    name: str = "SerialSection"
    required_keys: Tuple[str] = REQUIRED_KEYS["SerialSection"]
    patterns = None
    protocol = None
    point = None
    # ref: "serial-liftout section" https://www.nature.com/articles/s41592-023-02113-5

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemRectangleSettings]:
        """Calculate the serial liftout sectioning milling patterns"""

        check_keys(protocol, self.required_keys)

        # TODO: make side patterns optional

        section_thickness = protocol["section_thickness"]
        section_width = protocol["section_width"]
        section_depth = protocol["section_depth"]
        side_width = protocol["side_width"]
        side_height = protocol.get("side_height",  0)
        side_depth = protocol["side_depth"]
        inverted = protocol.get("inverted", False)
        use_side_patterns = protocol.get("use_side_patterns", True)

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
        
        self.patterns = [section_pattern]

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

            self.patterns += [left_side_pattern, right_side_pattern, 
                            left_side_pattern_vertical, 
                            right_side_pattern_vertical]
            
        self.protocol = protocol
        self.point = point

        return self.patterns

@dataclass
class RectangleOffsetPattern(BasePattern):
    name: str = "RectangleOffset"
    required_keys: Tuple[str] = REQUIRED_KEYS["RectangleOffset"]
    patterns = None
    protocol = None
    point = None

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> List[FibsemRectangleSettings]:
        check_keys(protocol, self.required_keys)

        width = protocol["width"]
        height = protocol["height"]
        depth = protocol["depth"]
        offset = protocol["offset"]
        scan_direction = protocol.get("scan_direction", "TopToBottom")
        cross_section = CrossSectionPattern[protocol.get("cross_section", "Rectangle")]
        inverted = protocol.get("inverted", False)

        offset = offset + height / 2
        if inverted:
            offset = -offset
            
        center_y = point.y + offset

        pattern = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=point.x,
            centre_y=center_y,
            scan_direction=scan_direction,
            cross_section = cross_section,
        )

        self.patterns = [pattern]
        self.protocol = protocol
        self.point = point
        return self.patterns

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
        from fibsem import constants
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
        use_cleaning_cross_section = protocol.get("cleaning_cross_section", False)

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
            cleaning_cross_section=use_cleaning_cross_section,
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
            cleaning_cross_section=use_cleaning_cross_section,
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
            cleaning_cross_section=True,
            scan_direction="TopToBottom",
        )

        right_pattern_settings = FibsemRectangleSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=point.x + protocol["distance"],
            centre_y=point.y,
            cleaning_cross_section=True,
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
                cleaning_cross_section=False,
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
            cleaning_cross_section=False,
            scan_direction="TopToBottom",
            cross_section=cross_section
        )

        bottom_vertical_pattern = FibsemRectangleSettings(
            width=vwidth,
            height=vheight,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y + distance / 2 + vheight / 2 - hheight / 2,
            cleaning_cross_section=False,
            scan_direction="BottomToTop",
            cross_section=cross_section
        )

        top_horizontal_pattern = FibsemRectangleSettings(
            width=hwidth,
            height=hheight,
            depth=depth,
            centre_x=point.x + (hwidth / 2 + vwidth / 2) * inverted,
            centre_y=point.y - distance / 2,
            cleaning_cross_section=False,
            scan_direction="TopToBottom",
            cross_section=cross_section
        )

        bottom_horizontal_pattern = FibsemRectangleSettings(
            width=hwidth,
            height=hheight,
            depth=depth,
            centre_x=point.x + (hwidth / 2 + vwidth / 2) * inverted,
            centre_y=point.y + distance / 2,
            cleaning_cross_section=False,
            scan_direction="BottomToTop",
            cross_section=cross_section
        )

        centre_vertical_pattern = FibsemRectangleSettings(
            width=vwidth,
            height=distance + hheight,
            depth=depth,
            centre_x=point.x + (hwidth + vwidth) * inverted,
            centre_y=point.y,
            cleaning_cross_section=False,
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
            cleaning_cross_section=False,
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
                cleaning_cross_section=False,
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
                cleaning_cross_section=False,
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
    RectangleOffsetPattern,
    UndercutPattern,
    FiducialPattern,
    ArrayPattern,
    MicroExpansionPattern,
    WaffleNotchPattern,
    CloverPattern,
    TriForcePattern,
    BitmapPattern,
    AnnulusPattern,
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
        cleaning_cross_section=False,
        scan_direction="LeftToRight",
    )

    right_pattern = FibsemRectangleSettings(
        width=width,
        height=height,
        depth=depth,
        rotation=np.deg2rad(angle),
        centre_x=point.x + h_offset,
        centre_y=point.y,
        cleaning_cross_section=False,
        scan_direction="RightToLeft",
    )

    bottom_pattern = FibsemRectangleSettings(
        width=width,
        height=height,
        depth=depth,
        rotation=np.deg2rad(90),
        centre_x=point.x,
        centre_y=point.y - height / 2,
        cleaning_cross_section=False,
        scan_direction="BottomToTop",
    )

    return [left_pattern, right_pattern, bottom_pattern]

# TODO: move to structures
@dataclass
class FibsemMillingStage:
    name: str = "Milling Stage"
    num: int = 0
    milling: FibsemMillingSettings = FibsemMillingSettings()
    pattern: BasePattern = get_pattern("Rectangle")

    def to_dict(self):
        return {
            "name": self.name,
            "num": self.num,
            "milling": self.milling.to_dict(),
            "pattern": self.pattern.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            name=data["name"],
            num=data["num"],
            milling=FibsemMillingSettings.from_dict(data["milling"]),
            pattern=get_pattern(data["pattern"]["name"]).from_dict(data["pattern"]),
        )


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


def _get_stage(key, protocol: dict, point: Point = Point(), i: int = 0) -> FibsemMillingStage:
    pattern = _get_pattern(key, protocol, point=point)
    mill_settings = FibsemMillingSettings(
        milling_current=protocol["milling_current"], 
        hfw=float(protocol["hfw"]),
        application_file=protocol.get("application_file", "Si"),
        preset=protocol.get("preset", "30 keV; 20 nA"),
        patterning_mode=protocol.get("patterning_mode", "Serial"))

    # milling stage name
    name = protocol.get("name", f"{key.title()} {i+1:02d}")

    stage = FibsemMillingStage(
        name=name, num=i, milling=mill_settings, pattern=pattern
    )
    return stage




def get_milling_stages(key, protocol, point: Union[Point, List[Point]] = Point()):
    
    # TODO: maybe add support for defining point per stages?

    # convert point to list of points, same length as stages

    if "stages" in protocol[key]:
        stages = []
        for i, pstage in enumerate(protocol[key]["stages"]):
            
            if isinstance(point, list):
                pt = point[i]
            else:
                pt = point

            stages.append(_get_stage(key, pstage, point=pt, i=i))
    else:
        stages = [_get_stage(key, protocol[key], point=point)]
    return stages


def get_protocol_from_stages(stages: list):
    protocol = {}  
    protocol["stages"] = []

    if not isinstance(stages, list):
        stages = [stages]

    for stage in stages:
        # join two dicts union
        ddict = {**stage.to_dict()["milling"], 
            **stage.to_dict()["pattern"]["protocol"], 
            "type": stage.to_dict()["pattern"]["name"],
            "name": stage.name}
        protocol["stages"].append(deepcopy(ddict))

    return protocol