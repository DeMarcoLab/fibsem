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

from abc import ABC, abstractmethod

class Pattern(ABC):
    @abstractmethod
    def __init__(self, pattern_type: MillingPattern):
        self.pattern_type = pattern_type

    @abstractmethod
    def define(self, protocol:dict) -> list[FibsemPatternSettings]:
        pass

    @abstractmethod
    def draw(self, microscope: FibsemMicroscope, protocol:dict, point: Point = Point()):
        pass

class RectanglePattern(Pattern):

    def __init__(self, pattern_type: MillingPattern):
        super().__init__(pattern_type)

    def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
        return FibsemPatternSettings.__from_dict__(protocol)

    def draw(self, microscope: FibsemMicroscope, protocol: dict, point: Point = Point()):
        return microscope.draw_rectangle(self.define(protocol, point))


class LinePattern(Pattern):
    
    def __init__(self, pattern_type: MillingPattern):
        super().__init__(pattern_type)

    def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
        return [FibsemPatternSettings.__from_dict__(protocol)]

    def draw(self, microscope: FibsemMicroscope, protocol: dict, point: Point = Point()):
        
        return microscope.draw_line(self.define(protocol, point)[0])

class CirclePattern(Pattern):
    
    def __init__(self, pattern_type: MillingPattern):
        super().__init__(pattern_type)

    def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
        
        return [FibsemPatternSettings.__from_dict__(protocol)]

    def draw(self, microscope: FibsemMicroscope, protocol:dict, point: Point = Point()):
        
        microscope.draw_circle(self.define(protocol, point)[0])


class TrenchPattern(Pattern):
    
        def __init__(self, pattern_type: MillingPattern):
            super().__init__(pattern_type)
    
        def define(self, protocol: dict, point: Point = Point()) -> list[FibsemPatternSettings]:
            
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

class HorseshoePattern(Pattern):
    
        def __init__(self, pattern_type: MillingPattern):
            super().__init__(pattern_type)
    
        def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
            """Calculate the trench milling patterns"""

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

class FiducialPattern(Pattern):
    
        def __init__(self, pattern_type: MillingPattern):
            super().__init__(pattern_type)
    
        def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
            import numpy as np

            left_pattern = FibsemPatternSettings.__from_dict__(protocol)
            left_pattern.rotation = np.deg2rad(45)
            right_pattern = FibsemPatternSettings.__from_dict__(protocol)
            right_pattern.rotation = left_pattern + np.deg2rad(90)

            return [left_pattern, right_pattern]

        def draw(self, microscope: FibsemMicroscope, protocol:dict, point: Point = Point()):
            
            for pattern in self.define(protocol, point):
                microscope.draw_rectangle(pattern)

class UndercutPattern(Pattern):
    
        def __init__(self, pattern_type: MillingPattern):
            super().__init__(pattern_type)
    
        def define(self, protocol:dict, point: Point = Point()) -> list[FibsemPatternSettings]:
            pass

        def draw(self, microscope: FibsemMicroscope, protocol:dict, point: Point = Point()):
            pass

class MicroExpansionPattern(Pattern):
    
    def __init__(self, pattern_type: MillingPattern):
        super().__init__(pattern_type)

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
        depth = protocol["milling_depth"] # lamella milling depth

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






