from dataclasses import dataclass
from enum import Enum

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemPatternSettings, FibsemPattern, Point

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
# TODO: define all the patterns
# TODO: unify the draw method
# TODO: fix the ui install / test in ui
# TODO: map the patterns to types for liftout? (separate issue)
# TODO: i think these should be abstractdataclasses insstead?
# Maybe strategy pattern? instead of ABC? drawStrategy? defineStrategy?

from abc import ABC, abstractmethod 

def check_keys(protocol: dict, required_keys: list[str]) -> bool:
    return all([k in protocol.keys() for k in required_keys])

@dataclass
class BasePattern(ABC):
    name: str = "BasePattern"
    required_keys: list[str] = []

    @abstractmethod
    def define(self, protocol:dict) -> list[FibsemPatternSettings]:
        pass

    @abstractmethod
    def draw(self, microscope: FibsemMicroscope, protocol:dict, point: Point = Point()):
        pass

@dataclass
class RectanglePattern(BasePattern):
    name: str = "Rectangle"
    required_keys: list[str] = ["width", "height", "depth"]



    def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["pattern"]  = "Rectangle" # redundant now
        return [FibsemPatternSettings.__from_dict__(protocol)]

    def draw(self, microscope: FibsemMicroscope, protocol: dict, point: Point = Point()):
        return microscope.draw_rectangle(self.define(protocol, point)[0])

@dataclass
class LinePattern(BasePattern):
    name: str = "Line"
    required_keys: list[str] = ["length", "depth"]
    
    def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
        protocol["pattern"]  = "Line" # redundant now
        return [FibsemPatternSettings.__from_dict__(protocol)]

    def draw(self, microscope: FibsemMicroscope, protocol: dict, point: Point = Point()):
        
        return microscope.draw_line(self.define(protocol, point)[0])

@dataclass
class CirclePattern(BasePattern):
    name: str = "Circle"
    required_keys: list[str] = ["radius", "depth"]

    def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
        
        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["pattern"]  = "Circle" # redundant now
        return [FibsemPatternSettings.__from_dict__(protocol)]

    def draw(self, microscope: FibsemMicroscope, protocol:dict, point: Point = Point()):
        
        microscope.draw_circle(self.define(protocol, point)[0])

@dataclass
class TrenchPattern(BasePattern):
    name: str = "Trench"
    required_keys: list[str] = ["lamella_width", "lamella_height", 
                   "trench_height", "size_ratio", 
                   "offset", "milling_depth"]

    def define(self, protocol: dict, point: Point = Point()) -> list[FibsemPatternSettings]:
        
        check_keys(protocol, self.required_keys)

        lamella_width = protocol["lamella_width"]
        lamella_height = protocol["lamella_height"]
        trench_height = protocol["trench_height"]
        upper_trench_height = trench_height / max(protocol["size_ratio"], 1.0)
        offset = protocol["offset"]
        milling_depth = protocol["milling_depth"]

        centre_upper_y = point.y + (lamella_height / 2 + upper_trench_height / 2 + offset)
        centre_lower_y = point.y - (lamella_height / 2 + trench_height / 2 + offset)

        # mill settings
        lower_pattern_settings = FibsemPatternSettings(
            width=lamella_width,
            height=trench_height,
            depth=milling_depth,
            centre_x=point.x,
            centre_y=centre_lower_y,
            cleaning_cross_section=True,
            scan_direction="BottomToTop",
        )

        upper_pattern_settings = FibsemPatternSettings(
            width=lamella_width,
            height=upper_trench_height,
            depth=milling_depth,
            centre_x=point.x,
            centre_y=centre_upper_y,
            cleaning_cross_section=True,
            scan_direction="TopToBottom",
        )

        return [lower_pattern_settings, upper_pattern_settings]
    
    def draw(self, microscope: FibsemMicroscope, protocol: dict, point: Point = Point()):
        """Calculate the trench milling patterns"""

        patterns = self.define(protocol, point)
        for pattern in patterns:
            microscope.draw_rectangle(pattern) # probs a better way than individually doing this? map draw method to pattern type?

@dataclass
class HorseshoePattern(BasePattern):
    name: str = "Horseshoe"
    required_keys: list[str] = ["lamella_width", "lamella_height", 
                    "trench_height", "size_ratio", 
                    "side_width",
                    "offset", "milling_depth"]

    # TODO:ref: "horseshoe" terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14

    def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
        """Calculate the trench milling patterns"""

        check_keys(protocol, self.required_keys)
        
        lamella_width = protocol["lamella_width"]
        lamella_height = protocol["lamella_height"]
        trench_height = protocol["trench_height"]
        upper_trench_height = trench_height / max(protocol["size_ratio"], 1.0)
        offset = protocol["offset"]
        milling_depth = protocol["milling_depth"]

        centre_upper_y = point.y + (lamella_height / 2 + upper_trench_height / 2 + offset)
        centre_lower_y = point.y - (lamella_height / 2 + trench_height / 2 + offset)

        lower_pattern = FibsemPatternSettings(
            width=lamella_width,
            height=trench_height,
            depth=milling_depth,
            centre_x=point.x,
            centre_y=centre_lower_y,
            cleaning_cross_section=True,
            scan_direction="BottomToTop",
        )

        upper_pattern = FibsemPatternSettings(
            width=lamella_width,
            height=upper_trench_height,
            depth=milling_depth,
            centre_x=point.x,
            centre_y=centre_upper_y,
            cleaning_cross_section=True,
            scan_direction="TopToBottom",
        )

        side_pattern = FibsemPatternSettings(
            width=protocol["side_width"],
            height=lamella_height + offset,
            depth=milling_depth,
            centre_x=point.x - (lamella_width / 2 - protocol["side_width"] / 2),
            centre_y=point.y,
            cleaning_cross_section=True,
            scan_direction="TopToBottom",
        )
        
        return [lower_pattern, upper_pattern, side_pattern]

    def draw(self, microscope: FibsemMicroscope, protocol:dict, point: Point = Point()):
        
        patterns = self.define(protocol, point)
        for pattern in patterns:
            microscope.draw_rectangle(pattern)

@dataclass
class FiducialPattern(BasePattern):
    name: str = "Fiducial"
    required_keys: list[str] = ["rotation", "depth", "width"]

    def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
        import numpy as np

        # ?
        protocol["centre_x"] = point.x
        protocol["centre_y"] = point.y
        protocol["pattern"] = "Rectangle"

        left_pattern = FibsemPatternSettings.__from_dict__(protocol)
        left_pattern.rotation = np.deg2rad(45)
        right_pattern = FibsemPatternSettings.__from_dict__(protocol)
        right_pattern.rotation = left_pattern + np.deg2rad(90)

        return [left_pattern, right_pattern]

    def draw(self, microscope: FibsemMicroscope, protocol:dict, point: Point = Point()):
        
        for pattern in self.define(protocol, point):
            microscope.draw_rectangle(pattern)

@dataclass
class UndercutPattern(BasePattern):
    name: str = "Undercut"
    required_keys: list[str] = ["width", "depth", "rhs_height", "h_offset", "offset"]
    
    def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
        
        # jcut_lhs_height = protocol["lhs_height"]
        jcut_rhs_height = protocol["rhs_height"]
        jcut_lamella_height = protocol["lamella_height"]
        jcut_width = protocol["width"]
        jcut_trench_thickness = protocol["trench_width"]
        # jcut_lhs_trench_thickness = protocol["lhs_trench_width"]
        # jcut_lhs_offset = protocol["lhs_offset"]
        jcut_milling_depth = protocol["depth"]
        jcut_h_offset = protocol["h_offset"]

        jcut_half_width = jcut_width - jcut_trench_thickness / 2
        jcut_half_height = jcut_lamella_height / 2

        use_cleaning_cross_section = protocol.get("cleaning_cross_section", True)

        # top_jcut
        jcut_top_centre_x = point.x + jcut_width / 2 - jcut_h_offset
        jcut_top_centre_y = point.y + jcut_lamella_height
        jcut_top_width = jcut_width
        jcut_top_height = jcut_trench_thickness
        jcut_top_depth = jcut_milling_depth
        
        top_pattern = FibsemPatternSettings(
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
        #     depth=jcut_milling_depth,
        # )  # depth

        # rhs jcut
        jcut_rhs_centre_x = point.x + jcut_half_width - jcut_h_offset
        jcut_rhs_centre_y = point.y + jcut_half_height - (jcut_rhs_height / 2 - jcut_half_height) + jcut_trench_thickness / 2
        jcut_rhs_width = jcut_trench_thickness
        jcut_rhs_height = jcut_rhs_height
        jcut_rhs_depth = jcut_milling_depth

        rhs_pattern = FibsemPatternSettings(
            width=jcut_rhs_width,
            height=jcut_rhs_height,
            depth=jcut_rhs_depth,
            centre_x=jcut_rhs_centre_x,
            centre_y=jcut_rhs_centre_y,
            cleaning_cross_section=use_cleaning_cross_section,
            scan_direction="TopToBottom",
        )


        return [top_pattern, rhs_pattern]


    def draw(self, microscope: FibsemMicroscope, protocol:dict, point: Point = Point()):
        
        for pattern in self.define(protocol, point):
            microscope.draw_rectangle(pattern)

@dataclass
class MicroExpansionPattern(BasePattern):
    name: str = "MicroExpansion"
    required_keys: list[str] = ["width", "height", "depth", "distance", "lamella_width"]

    def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
        """
        Draw the microexpansion joints for stress relief of lamella.

        Args:
            microscope (FibsemMicroscope): OpenFIBSEM microscope instance
            protocol (dict): Contains a dictionary of the necessary values for drawing the joints.
            protocol (dict): Lamella protocol

        Returns:
            patterns: list[FibsemPatternSettings]
        """
        width = protocol["width"]
        height = protocol["height"]
        depth = protocol["depth"] # lamella milling depth

        left_pattern_settings = FibsemPatternSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=point.x
            - protocol["lamella_width"] / 2 # lamella property
            - protocol["distance"],
            centre_y=point.y,
            cleaning_cross_section=True,
            scan_direction="LeftToRight",
        )

        right_pattern_settings = FibsemPatternSettings(
            width=width,
            height=height,
            depth=depth,
            centre_x=point.x
            + protocol["lamella_width"] / 2
            + protocol["distance"],
            centre_y=point.y,
            cleaning_cross_section=True,
            scan_direction="RightToLeft",
        )

        return [left_pattern_settings, right_pattern_settings]

    def draw(self, microscope: FibsemMicroscope, protocol:dict, point: Point = Point()):
        
        patterns = self.define(protocol, point)
        for pattern in patterns:
            microscope.draw_rectangle(pattern)


@dataclass
class SpotWeldPattern(BasePattern):
    name: str = "SpotWeld"
    required_keys: list[str] = ["width", "height", "depth", "distance", "number"]
    # ref: spotweld terminology https://www.researchgate.net/publication/351737991_A_Modular_Platform_for_Streamlining_Automated_Cryo-FIB_Workflows#pf14

    def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:

        width = protocol["width"]
        height = protocol["height"]
        depth = protocol["depth"]
        distance = protocol["distance"]
        n_patterns = protocol["number"]

        patterns = []
        for i in range(n_patterns):
            pattern_settings = FibsemPatternSettings(
                width=width,
                height=height,
                depth=depth,
                centre_x=point.x + (i - (n_patterns - 1) / 2) * distance,
                centre_y=point.y,
                cleaning_cross_section=True,
                scan_direction="LeftToRight",
            )
            patterns.append(pattern_settings)

        return patterns

    def draw(self, microscope: FibsemMicroscope, protocol:dict, point: Point = Point()):
            
            patterns = self.define(protocol, point)
            for pattern in patterns:
                microscope.draw_rectangle(pattern)



__PATTERNS__ = [RectanglePattern, LinePattern, CirclePattern, TrenchPattern, HorseshoePattern, UndercutPattern, FiducialPattern, MicroExpansionPattern, SpotWeldPattern]
