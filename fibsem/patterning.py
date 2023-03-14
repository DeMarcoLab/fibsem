from dataclasses import dataclass
from enum import Enum

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemPatternSettings, FibsemPattern, Point
import logging

class MillingPattern(Enum):
    Trench = 1
    JCut = 2
    Sever = 3
    Weld = 4
    Cut = 5
    Sharpen = 6
    Thin = 7
    Polish = 8
    Flatten = 9
    Fiducial = 10
    Horseshoe = 11
    Line = 12
    Rectangle = 13


# @patrickcleeve2 March 7
# TODO: unify the draw method
# TODO: fix the ui install / test in ui
# TODO: map the patterns to types for liftout? (separate issue)
# Maybe strategy pattern? instead of ABC? drawStrategy? defineStrategy?

from abc import ABC, abstractmethod


def check_keys(protocol: dict, required_keys: list[str]) -> bool:
    return all([k in protocol.keys() for k in required_keys])


@dataclass
class BasePattern(ABC):
    name: str = "BasePattern"
    required_keys: tuple[str] = ()

    @abstractmethod
    def define(self, protocol: dict, point: Point = Point()) -> list[FibsemPatternSettings]:
        pass

@dataclass
class RectanglePattern(BasePattern):
    name: str = "Rectangle"
    required_keys: tuple[str] = ("width", "height", "depth", "rotation")

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> list[FibsemPatternSettings]:
        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["pattern"] = "Rectangle"  # redundant now
        protocol["cleaning_cross_section"] = protocol.get("cleaning_cross_section", False)
        protocol["scan_direction"] = protocol.get("scan_direction", "TopToBottom")
        self.patterns = [FibsemPatternSettings.__from_dict__(protocol)]

        return self.patterns 


@dataclass
class LinePattern(BasePattern):
    name: str = "Line"
    required_keys: tuple[str] = ("start_x", "end_x", "start_y", "end_y", "depth")

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> list[FibsemPatternSettings]:
        protocol["pattern"] = "Line"  # redundant now
        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["cleaning_cross_section"] = protocol.get("cleaning_cross_section", False)
        protocol["scan_direction"] = protocol.get("scan_direction", "TopToBottom")
        self.patterns = [FibsemPatternSettings.__from_dict__(protocol)]
        return self.patterns


@dataclass
class CirclePattern(BasePattern):
    name: str = "Circle"
    required_keys: tuple[str] = ("radius", "depth")

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> list[FibsemPatternSettings]:

        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["pattern"] = "Circle"  # redundant now
        self.patterns = [FibsemPatternSettings.__from_dict__(protocol)] 
        return self.patterns


@dataclass
class TrenchPattern(BasePattern):
    name: str = "Trench"
    required_keys: tuple[str] = (
        "lamella_width",
        "lamella_height",
        "trench_height",
        "size_ratio",
        "offset",
        "depth",
    )

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> list[FibsemPatternSettings]:

        check_keys(protocol, self.required_keys)

        lamella_width = protocol["lamella_width"]
        lamella_height = protocol["lamella_height"]
        trench_height = protocol["trench_height"]
        upper_trench_height = trench_height / max(protocol["size_ratio"], 1.0)
        offset = protocol["offset"]
        depth = protocol["depth"]

        centre_upper_y = point.y + (
            lamella_height / 2 + upper_trench_height / 2 + offset
        )
        centre_lower_y = point.y - (lamella_height / 2 + trench_height / 2 + offset)

        # mill settings
        lower_pattern_settings = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=lamella_width,
            height=trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_lower_y,
            cleaning_cross_section=True,
            scan_direction="BottomToTop",
        )

        upper_pattern_settings = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=lamella_width,
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_upper_y,
            cleaning_cross_section=True,
            scan_direction="TopToBottom",
        )

        self.patterns = [lower_pattern_settings, upper_pattern_settings]    
        return self.patterns


@dataclass
class HorseshoePattern(BasePattern):
    name: str = "Horseshoe"
    required_keys: tuple[str] = (
        "lamella_width",
        "lamella_height",
        "trench_height",
        "size_ratio",
        "offset",
        "side_offset",
        "side_width",
        "depth",
    )

    # TODO:ref: "horseshoe" terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> list[FibsemPatternSettings]:
        """Calculate the trench milling patterns"""

        check_keys(protocol, self.required_keys)

        lamella_width = protocol["lamella_width"]
        lamella_height = protocol["lamella_height"]
        trench_height = protocol["trench_height"]
        upper_trench_height = trench_height / max(protocol["size_ratio"], 1.0)
        offset = protocol["offset"]
        depth = protocol["depth"]

        centre_upper_y = point.y + (
            lamella_height / 2 + upper_trench_height / 2 + offset
        )
        centre_lower_y = point.y - (lamella_height / 2 + trench_height / 2 + offset)

        lower_pattern = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=lamella_width,
            height=trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_lower_y,
            cleaning_cross_section=True,
            scan_direction="BottomToTop",
        )

        upper_pattern = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=lamella_width,
            height=upper_trench_height,
            depth=depth,
            centre_x=point.x,
            centre_y=centre_upper_y,
            cleaning_cross_section=True,
            scan_direction="TopToBottom",
        )

        side_pattern = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=protocol["side_width"],
            height=lamella_height + offset,
            depth=depth,
            centre_x=point.x - protocol["side_offset"] + (lamella_width / 2 - protocol["side_width"] / 2) ,
            centre_y=point.y,
            cleaning_cross_section=True,
            scan_direction="TopToBottom",
        )

        self.patterns = [lower_pattern, upper_pattern, side_pattern]
        return self.patterns

@dataclass
class FiducialPattern(BasePattern):
    name: str = "Fiducial"
    required_keys: tuple[str] = ("height", "width", "depth", "rotation")

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> list[FibsemPatternSettings]:
        import numpy as np

        # ?
        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["pattern"] = "Rectangle"
        protocol["cleaning_cross_section"] = protocol.get("cleaning_cross_section", False)
        protocol["scan_direction"] = protocol.get("scan_direction", "TopToBottom")

        left_pattern = FibsemPatternSettings.__from_dict__(protocol)
        left_pattern.rotation = np.deg2rad(45)
        right_pattern = FibsemPatternSettings.__from_dict__(protocol)
        right_pattern.rotation = left_pattern.rotation + np.deg2rad(90)

        self.patterns = [left_pattern, right_pattern]
        return self.patterns


@dataclass
class UndercutPattern(BasePattern):
    name: str = "Undercut"
    required_keys: tuple[str] = ("height", "width", "depth", "trench_width", "rhs_height", "h_offset", "offset")

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> list[FibsemPatternSettings]:

        check_keys(protocol, self.required_keys)

        # jcut_lhs_height = protocol["lhs_height"]
        jcut_rhs_height = protocol["rhs_height"]
        jcut_lamella_height = protocol["height"]
        jcut_width = protocol["width"]
        jcut_trench_thickness = protocol["trench_width"]
        # jcut_lhs_trench_thickness = protocol["lhs_trench_width"]
        # jcut_lhs_offset = protocol["lhs_offset"]
        jcut_depth = protocol["depth"]
        jcut_h_offset = protocol["h_offset"]

        jcut_half_width = jcut_width - jcut_trench_thickness / 2
        jcut_half_height = jcut_lamella_height / 2

        use_cleaning_cross_section = protocol.get("cleaning_cross_section", True)

        # top_jcut
        jcut_top_centre_x = point.x + jcut_width / 2 - jcut_h_offset
        jcut_top_centre_y = point.y + jcut_lamella_height
        jcut_top_width = jcut_width
        jcut_top_height = jcut_trench_thickness
        jcut_top_depth = jcut_depth

        top_pattern = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=jcut_top_width,
            height=jcut_top_height,
            depth=jcut_top_depth,
            centre_x=jcut_top_centre_x,
            centre_y=jcut_top_centre_y,
            cleaning_cross_section=use_cleaning_cross_section,
            scan_direction="TopToBottom",
        )

        # lhs_jcut
        # jcut_lhs = microscope.patterning.create_rectangle(
        #     center_x=point.x - jcut_half_width - jcut_lhs_offset,
        #     center_y=point.y + jcut_half_height - (jcut_lhs_height / 2 - jcut_half_height),
        #     width=jcut_lhs_trench_thickness,
        #     height=jcut_lhs_height,
        #     depth=jcut_depth,
        # )  # depth

        # rhs jcut
        jcut_rhs_centre_x = point.x + jcut_half_width - jcut_h_offset
        jcut_rhs_centre_y = (
            point.y
            + jcut_half_height
            - (jcut_rhs_height / 2 - jcut_half_height)
            + jcut_trench_thickness / 2
        )
        jcut_rhs_width = jcut_trench_thickness
        jcut_rhs_height = jcut_rhs_height
        jcut_rhs_depth = jcut_depth

        rhs_pattern = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=jcut_rhs_width,
            height=jcut_rhs_height,
            depth=jcut_rhs_depth,
            centre_x=jcut_rhs_centre_x,
            centre_y=jcut_rhs_centre_y,
            cleaning_cross_section=use_cleaning_cross_section,
            scan_direction="TopToBottom",
        )

        self.patterns = [top_pattern, rhs_pattern]
        return self.patterns


@dataclass
class MicroExpansionPattern(BasePattern):
    name: str = "MicroExpansion"
    required_keys: tuple[str] = (
        "height",
        "width",
        "depth",
        "distance",
        "lamella_width",
    )
    # ref: https://www.nature.com/articles/s41467-022-29501-3
    def define(
        self, protocol: dict, point: Point = Point()
    ) -> list[FibsemPatternSettings]:
        """
        Draw the microexpansion joints for stress relief of lamella.

        Args:
            microscope (FibsemMicroscope): OpenFIBSEM microscope instance
            protocol (dict): Contains a dictionary of the necessary values for drawing the joints.
            protocol (dict): Lamella protocol

        Returns:
            patterns: list[FibsemPatternSettings]
        """
        check_keys(protocol, self.required_keys)

        width = protocol["width"]
        height = protocol["height"]
        depth = protocol["depth"]  # lamella milling depth

        left_pattern_settings = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=width,
            height=height,
            depth=depth,
            centre_x=point.x
            - protocol["lamella_width"] / 2  # lamella property
            - protocol["distance"],
            centre_y=point.y,
            cleaning_cross_section=True,
            scan_direction="LeftToRight",
        )

        right_pattern_settings = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=width,
            height=height,
            depth=depth,
            centre_x=point.x + protocol["lamella_width"] / 2 + protocol["distance"],
            centre_y=point.y,
            cleaning_cross_section=True,
            scan_direction="RightToLeft",
        )

        self.patterns = [left_pattern_settings, right_pattern_settings]
        return self.patterns

@dataclass
class SpotWeldPattern(BasePattern):
    name: str = "SpotWeld"
    required_keys: tuple[str] = ("height","width" , "depth", "distance", "number")
    # ref: spotweld terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14

    def define(
        self, protocol: dict, point: Point = Point()
    ) -> list[FibsemPatternSettings]:

        check_keys(protocol, self.required_keys)
        width = protocol["width"]
        height = protocol["height"]
        depth = protocol["depth"]
        distance = protocol["distance"]
        n_patterns = int(protocol["number"])

        patterns = []
        for i in range(n_patterns):
            pattern_settings = FibsemPatternSettings(
                pattern=FibsemPattern.Rectangle,
                width=width,
                height=height,
                depth=depth,
                centre_x=point.x ,
                centre_y=point.y + (i - (n_patterns - 1) / 2) * distance,
                cleaning_cross_section=True,
                scan_direction="LeftToRight",
            )
            patterns.append(pattern_settings)

        self.patterns = patterns
        return self.patterns

@dataclass
class WaffleNotchPattern(BasePattern):
    name: str = "WaffleNotch"
    required_keys: tuple[str] = (
            "vheight",
            "vwidth",
            "hheight",
            "hwidth",
            "depth",
            "distance",
            "lamella_width",
        )

    # ref: https://www.nature.com/articles/s41467-022-29501-3

    def define(self, protocol: dict, point: Point = Point() ) -> list[FibsemPatternSettings]:

        check_keys(protocol, self.required_keys)

        vwidth = protocol["vwidth"]
        vheight = protocol["vheight"]
        hwidth = protocol["hwidth"]
        hheight = protocol["hheight"]
        depth = protocol["depth"]
        distance = protocol["distance"]

        # five patterns
        top_vertical_pattern = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=vwidth,
            height=vheight,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y - distance/2 - vheight/2 + hheight/2,
            cleaning_cross_section=False,
            scan_direction="TopToBottom",
        )

        bottom_vertical_pattern = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=vwidth,
            height=vheight,
            depth=depth,
            centre_x=point.x,
            centre_y=point.y + distance/2 + vheight/2 - hheight/2,
            cleaning_cross_section=False,
            scan_direction="BottomToTop",
        )


        top_horizontal_pattern = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=hwidth,
            height=hheight,
            depth=depth,
            centre_x=point.x + hwidth/2 + vwidth/2,
            centre_y=point.y - distance/2,
            cleaning_cross_section=False,
            scan_direction="TopToBottom",
        )

        bottom_horizontal_pattern = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=hwidth,
            height=hheight,
            depth=depth,
            centre_x=point.x + hwidth/2 + vwidth/2,
            centre_y=point.y + distance/2,
            cleaning_cross_section=False,
            scan_direction="BottomToTop",
        )


        centre_vertical_pattern = FibsemPatternSettings(
            pattern=FibsemPattern.Rectangle,
            width=vwidth,
            height=distance + hheight,
            depth=depth,
            centre_x=point.x + hwidth + vwidth,
            centre_y=point.y,
            cleaning_cross_section=False,
            scan_direction="TopToBottom",
        )


        self.patterns = [top_vertical_pattern, bottom_vertical_pattern, top_horizontal_pattern, bottom_horizontal_pattern, centre_vertical_pattern]

        return self.patterns 


__PATTERNS__ = [
    RectanglePattern,
    LinePattern,
    CirclePattern,
    TrenchPattern,
    HorseshoePattern,
    UndercutPattern,
    FiducialPattern,
    SpotWeldPattern,
    MicroExpansionPattern,
    WaffleNotchPattern,
]
