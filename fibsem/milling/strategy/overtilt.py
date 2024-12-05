
from dataclasses import dataclass
import numpy as np

from fibsem.microscope import FibsemMicroscope
from fibsem import acquire, alignment
from fibsem.milling import draw_pattern, run_milling, setup_milling
from fibsem.milling.base import FibsemMillingStage, MillingStrategy
from fibsem.milling.patterning import patterns
from fibsem.structures import BeamType, FibsemImage, ImageSettings, FibsemStagePosition, FibsemRectangle
import os

DEFAULT_ALIGNMENT_AREA = {"left": 0.7, "top": 0.3, "width": 0.25, "height": 0.4}

@dataclass
class OvertiltTrenchMillingStrategy(MillingStrategy):
    """Overtilt milling strategy for trench milling"""
    name: str = "Overtilt Milling Strategy"

    def __init__(self, overtilt_deg: float = 1):
        self.overtilt_deg = overtilt_deg

    def run(self, microscope: FibsemMicroscope, stage: "FibsemMillingStage", asynch: bool = False,
        parent_ui = None,
        current_stage_index: int = 0,
        total_stages: int = 1,) -> None:

        """Mill a trench pattern with overtilt, 
        based on https://www.sciencedirect.com/science/article/abs/pii/S1047847716301514 and autolamella v1"""


        # assert pattern is TrenchPattern
        if not isinstance(stage.pattern, patterns.TrenchPattern):
            raise ValueError("Pattern must be TrenchPattern for overtilt milling")

        # save initial position
        initial_position = microscope.get_stage_position()
        overtilt_in_radians = np.deg2rad(self.overtilt_deg)

        # TODO: pass in image settings
        # TODO: attach image_settings to microscope? or get current settings?
        image_settings = ImageSettings(hfw=stage.pattern.protocol["hfw"], 
                                       dwell_time=1e-6, 
                                       resolution=[1536, 1024], beam_type=stage.milling.milling_channel)
        image_settings.reduced_area = FibsemRectangle.from_dict(DEFAULT_ALIGNMENT_AREA)
        image_settings.path = os.getcwd()
        image_settings.filename = f"ref_{stage.name}_overtilt_alignment"
        ref_image  = acquire.acquire_image(microscope, image_settings)


        for i, pattern in enumerate(stage.pattern.patterns):
            
            # TODO: validate which direction to tilt, including when combined with scan rotation
            scan_rotation = microscope.get("scan_rotation", stage.milling.milling_channel)
            # overtilt
            if i == 0:
                t = -overtilt_in_radians
            else:
                t = +overtilt_in_radians
            microscope.move_stage_relative(FibsemStagePosition(t=t))

            # beam alignment
            image_settings = ImageSettings.fromFibsemImage(ref_image)
            image_settings.filename = f"{stage.name}_overtilt_alignment_target_{i}"
            
            alignment.multi_step_alignment_v2(microscope=microscope, 
                                            ref_image=ref_image, 
                                            beam_type=stage.milling.milling_channel, 
                                            alignment_current=None,
                                            steps=3)

            # setup again to ensure we are milling at the correct current, cleared patterns
            setup_milling(microscope, stage.milling)

            # draw pattern
            draw_pattern(microscope, pattern)

            # run milling
            run_milling(microscope, stage.milling.milling_current, asynch=False)

            # return to initial position
            microscope.move_stage_absolute(initial_position)
