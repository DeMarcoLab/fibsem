from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field, fields, asdict
from typing import Dict, List, Tuple, Union, Any, Optional, Type, ClassVar, TypeVar, Generic

import numpy as np

from fibsem import constants
from fibsem.structures import (
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemRectangleSettings,
    Point,
    TFibsemPatternSettings,
)

TPattern = TypeVar("TPattern", bound="BasePattern")

# TODO: define the configuration for each key,
# e.g. 
# "width": {"type": "float", "min": 0, "max": 1000, "default": 100, "description": "Width of the rectangle"}
# "cross_section": {"type": "str", "options": [cs.name for cs in CrossSectionPattern], "default": "Rectangle", "description": "Cross section of the milling pattern"}

####### Combo Patterns

@dataclass
class BasePattern(ABC, Generic[TFibsemPatternSettings]):
    name: ClassVar[str] = field(init=False)
    point: Point = field(default_factory=Point)
    shapes: Optional[List[TFibsemPatternSettings]] = field(default=None, init=False)

    _advanced_attributes: ClassVar[Tuple[str, ...]] = ()

    @abstractmethod
    def define(self) -> List[TFibsemPatternSettings]:
        pass

    def to_dict(self) -> Dict[str, Any]:
        ddict = asdict(self)
        # Handle any special cases
        if "cross_section" in ddict:
            ddict["cross_section"] = ddict["cross_section"].name
        ddict["name"] = self.name
        del ddict["shapes"]
        return ddict

    @classmethod
    def from_dict(cls: Type[TPattern], ddict: Dict[str, Any]) -> TPattern:
        kwargs = {}
        for f in fields(cls):
            if f.name in ddict:
                kwargs[f.name] = ddict[f.name]

        # Construct objects
        point = kwargs.pop("point", None)
        if point is not None:
            kwargs["point"] = Point.from_dict(point)

        cross_section = kwargs.pop("cross_section", None)
        if cross_section is not None:
            kwargs["cross_section"] = CrossSectionPattern[cross_section]

        return cls(**kwargs)
                
    @property
    def required_attributes(self) -> Tuple[str, ...]:
        return tuple(f.name for f in fields(self) if f not in fields(BasePattern))
    
    @property
    def advanced_attributes(self) -> Tuple[str, ...]:
        return self._advanced_attributes

    @property
    def volume(self) -> float:
        # calculate the total volume of the milling pattern (sum of all shapes)
        return sum([shape.volume for shape in self.define()])

@dataclass
class BitmapPattern(BasePattern[FibsemBitmapSettings]):
    width: float = 10.0e-6
    height: float = 10.0e-6
    depth: float = 1.0e-6
    rotation: float = 0
    path: str = ""

    name: ClassVar[str] = "Bitmap"

    def define(self) -> List[FibsemBitmapSettings]:

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


@dataclass
class RectanglePattern(BasePattern[FibsemRectangleSettings]):
    width: float = 10.0e-6
    height: float = 5.0e-6
    depth: float = 1.0e-6
    rotation: float = 0
    time: float = 0  # means auto
    passes: int = 0  # means auto
    scan_direction: str = "TopToBottom"
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle

    name: ClassVar[str] = "Rectangle"
    # TODO: add for other patterns
    _advanced_attributes: ClassVar[Tuple[str, ...]] = ("time", "passes")

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


@dataclass
class LinePattern(BasePattern[FibsemLineSettings]):
    start_x: float = -10.0e-6
    end_x: float = 10.0e-6
    start_y: float = 0.0
    end_y: float = 0.0
    depth: float = 1.0e-6

    name: ClassVar[str] = "Line"

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
class CirclePattern(BasePattern[FibsemCircleSettings]):
    radius: float = 5.0e-6
    depth: float = 1.0e-6
    thickness: float = 0

    name: ClassVar[str] = "Circle"

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
class TrenchPattern(BasePattern[Union[FibsemRectangleSettings, FibsemCircleSettings]]):
    width: float = 10.0e-6
    depth: float = 2.0e-6
    spacing: float = 5.0e-6
    upper_trench_height: float = 5.0e-6
    lower_trench_height: float = 5.0e-6
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle
    time: float = 0.0  # means auto
    fillet: float = 0.0  # no fillet radius

    name: ClassVar[str] = "Trench"
    _advanced_attributes: ClassVar[Tuple[str, ...]] = ("time", "fillet")

    def define(self) -> List[Union[FibsemRectangleSettings, FibsemCircleSettings]]:

        point = self.point
        width = self.width
        spacing = self.spacing
        upper_trench_height = self.upper_trench_height
        lower_trench_height = self.lower_trench_height
        depth = self.depth
        cross_section = self.cross_section
        time = self.time
        fillet = self.fillet

        # calculate the centre of the upper and lower trench
        centre_lower_y = point.y - (spacing / 2 + lower_trench_height / 2)
        centre_upper_y = point.y + (spacing / 2 + upper_trench_height / 2)

        # fillet radius on the corners
        fillet = np.clip(fillet, 0, upper_trench_height / 2)
        if fillet > 0:
            width = max(0, width - 2 * fillet) # ensure width is not negative

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

        # add fillet to the corners
        if fillet > 0:            
            left_x_pos = point.x - width / 2
            right_x_pos = point.x + width / 2

            fillet_offset = 1.5
            lower_y_pos = centre_lower_y + lower_trench_height / 2 - fillet * fillet_offset
            top_y_pos = centre_upper_y - upper_trench_height / 2 + fillet * fillet_offset

            lower_left_fillet = FibsemCircleSettings(
                radius=fillet,
                depth=depth/2,
                centre_x=point.x - width / 2,
                centre_y=lower_y_pos,
            )
            lower_right_fillet = FibsemCircleSettings(
                radius=fillet,
                depth=depth/2,
                centre_x=point.x + width / 2,
                centre_y=lower_y_pos,
            )

            # fill the remaining space with rectangles
            lower_left_fill = FibsemRectangleSettings(
                width=fillet,
                height=lower_trench_height - fillet,
                depth=depth,
                centre_x=left_x_pos - fillet / 2,
                centre_y=centre_lower_y - fillet / 2,
                cross_section = cross_section,
                scan_direction="BottomToTop",

            )
            lower_right_fill = FibsemRectangleSettings(
                width=fillet,
                height=lower_trench_height - fillet,
                depth=depth,
                centre_x=right_x_pos + fillet / 2,
                centre_y=centre_lower_y - fillet / 2,
                cross_section = cross_section,
                scan_direction="BottomToTop",
            )

            top_left_fillet = FibsemCircleSettings(
                radius=fillet,
                depth=depth,
                centre_x=point.x - width / 2,
                centre_y=top_y_pos,
            )
            top_right_fillet = FibsemCircleSettings(
                radius=fillet,
                depth=depth,
                centre_x=point.x + width / 2,
                centre_y=top_y_pos,
            )

            top_left_fill = FibsemRectangleSettings(
                width=fillet,
                height=upper_trench_height - fillet,
                depth=depth,
                centre_x=left_x_pos - fillet / 2,
                centre_y=centre_upper_y + fillet / 2,
                cross_section = cross_section,
                scan_direction="TopToBottom",
            )
            top_right_fill = FibsemRectangleSettings(
                width=fillet,
                height=upper_trench_height - fillet,
                depth=depth,
                centre_x=right_x_pos + fillet / 2,
                centre_y=centre_upper_y + fillet / 2,
                cross_section = cross_section,
                scan_direction="TopToBottom",
            )

            self.shapes.extend([lower_left_fill, lower_right_fill, 
                                top_left_fill, top_right_fill, 
                                lower_left_fillet, lower_right_fillet, 
                                top_left_fillet, top_right_fillet])

        return self.shapes


@dataclass
class HorseshoePattern(BasePattern[FibsemRectangleSettings]):
    width: float = 40.0e-6
    upper_trench_height: float = 10.0e-6
    lower_trench_height: float = 10.0e-6
    spacing: float = 10.0e-6
    depth: float = 10.0e-6
    side_width: float = 5.0e-6
    inverted: bool = False
    scan_direction: str = "TopToBottom"
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle

    name: ClassVar[str] = "Horseshoe"
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
    

@dataclass
class HorseshoePatternVertical(BasePattern):
    width: float = 2.0e-05
    height: float = 5.0e-05
    side_trench_width: float = 5.0e-06
    top_trench_height: float = 10.0e-6
    depth: float = 4.0e-6
    scan_direction: str = "TopToBottom"
    inverted: bool = False
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle

    name: ClassVar[str] = "HorseshoeVertical"
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


@dataclass
class SerialSectionPattern(BasePattern[FibsemLineSettings]):
    section_thickness: float = 4.0e-6
    section_width: float = 50.0e-6
    section_depth: float = 20.0e-6
    side_width: float = 10.0e-6
    side_height: float = 10.0e-6
    side_depth: float = 40.0e-6
    inverted: bool = False
    use_side_patterns: bool = True

    name: ClassVar[str] = "SerialSection"
    # ref: "serial-liftout section" https://www.nature.com/articles/s41592-023-02113-5

    def define(self) -> List[FibsemLineSettings]:
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


@dataclass
class FiducialPattern(BasePattern[FibsemRectangleSettings]):
    width: float = 1.0e-6
    height: float = 10.0e-6
    depth: float = 5.0e-6
    rotation: float = 45.0
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle

    name: ClassVar[str] = "Fiducial"

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


@dataclass
class UndercutPattern(BasePattern[FibsemRectangleSettings]):
    width: float = 5.0e-6
    height: float = 10.0e-6
    depth: float = 10.0e-6
    trench_width: float = 2.0e-6
    rhs_height: float = 10.0e-6
    h_offset: float = 5.0e-6
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle

    name: ClassVar[str] = "Undercut"

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


@dataclass
class MicroExpansionPattern(BasePattern[FibsemRectangleSettings]):
    width: float = 0.5e-6
    height: float = 15.0e-6
    depth: float = 1.0e-6
    distance: float = 10.0e-6

    name: ClassVar[str] = "MicroExpansion"
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


@dataclass
class ArrayPattern(BasePattern[FibsemRectangleSettings]):
    width: float = 2.0e-6
    height: float = 2.0e-6
    depth: float = 5.0e-6
    n_columns: int = 5
    n_rows: int = 5
    pitch_vertical: float = 5.0e-6
    pitch_horizontal: float = 5.0e-6
    passes: int = 0  # means auto
    rotation: float = 0
    scan_direction: str = "TopToBottom"
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle

    name: ClassVar[str] = "ArrayPattern"
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


@dataclass
class WaffleNotchPattern(BasePattern[FibsemRectangleSettings]):
    vheight: float = 2.0e-6
    vwidth: float = 0.5e-6
    hheight: float = 0.5e-6
    hwidth: float = 2.0e-6
    depth: float = 1.0e-6
    distance: float = 2.0e-6
    inverted: bool = False
    cross_section: CrossSectionPattern = CrossSectionPattern.Rectangle

    name: ClassVar[str] = "WaffleNotch"
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


@dataclass
class CloverPattern(BasePattern[Union[FibsemCircleSettings, FibsemRectangleSettings]]):
    radius: float = 10.0e-6
    depth: float = 5.0e-6

    name: ClassVar[str] = "Clover"

    def define(self) -> List[Union[FibsemCircleSettings, FibsemRectangleSettings]]:
        
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


@dataclass
class TriForcePattern(BasePattern[FibsemRectangleSettings]):
    width: float = 1.0e-6
    height: float = 10.0e-6
    depth: float = 5.0e-6

    name: ClassVar[str] = "TriForce"

    def define(self) -> List[FibsemRectangleSettings]:
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


@dataclass
class TrapezoidPattern(BasePattern[FibsemRectangleSettings]):
    inner_width: float = 10.0e-6
    outer_width: float = 20.0e-6
    trench_height: float = 5.0e-6
    depth: float = 1.0e-6
    distance: float = 1.0e-6
    n_rectangles: int = 10
    overlap: float = 0.0

    name: ClassVar[str] = "Trapezoid"

    def define(self) -> List[FibsemRectangleSettings]:
        
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
            y = point.y + (i * self.trench_height * (1-overlap)) - distance - trench_height
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
            y = point.y - (i * self.trench_height * (1-overlap)) + distance + trench_height
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


# Pattern classes are now registered via the plugin system in __init__.py
# Legacy constants maintained for backwards compatibility
DEFAULT_MILLING_PATTERN = RectanglePattern
