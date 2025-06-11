
import logging
import os
from dataclasses import dataclass, field
from typing import Tuple, List
import numpy as np

from fibsem import acquire, alignment
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import draw_pattern, run_milling, setup_milling, finish_milling
from fibsem.milling.base import (FibsemMillingStage, MillingStrategy,
                                 MillingStrategyConfig)
from fibsem.milling.patterning.patterns2 import TrenchPattern
from fibsem.structures import (BeamType, FibsemImage, FibsemRectangle,
                               FibsemStagePosition, ImageSettings)


@dataclass
class OvertiltTrenchMillingConfig(MillingStrategyConfig):
    overtilt: float = 1
    resolution: List[int] = field(default_factory=lambda: [1536, 1024])


class OvertiltTrenchMillingStrategy(MillingStrategy[OvertiltTrenchMillingConfig]):
    """Overtilt milling strategy for trench milling"""
    name: str = "Overtilt"
    fullname: str = "Overtilt Trench Milling"
    config_class = OvertiltTrenchMillingConfig

    def run(self, microscope: FibsemMicroscope, stage: "FibsemMillingStage", asynch: bool = False,
        parent_ui = None) -> None:

        """Mill a trench pattern with overtilt, 
        based on https://www.sciencedirect.com/science/article/abs/pii/S1047847716301514 and autolamella v1"""
        logging.info(f"Running {self.fullname} for {stage.name}")

        # assert pattern is TrenchPattern
        if not isinstance(stage.pattern, TrenchPattern):
            raise ValueError("Pattern must be TrenchPattern for overtilt milling")

        # save initial position
        initial_position = microscope.get_stage_position()
        overtilt_in_radians = np.deg2rad(self.config.overtilt)

        # TODO: pass in image settings
        # TODO: attach image_settings to microscope? or get current settings?
        # TODO: use drift correction structure to re-align? once added to milling stage
        image_settings = ImageSettings(hfw=stage.milling.hfw,
                                       dwell_time=1e-6, 
                                       resolution=[1536, 1024], 
                                       beam_type=stage.milling.milling_channel)
        image_settings.reduced_area = stage.alignment.rect
        image_settings.path = os.getcwd()
        image_settings.filename = f"ref_{stage.name}_overtilt_alignment"
        ref_image = acquire.acquire_image(microscope, image_settings)

        # TODO: support rr
        for i, pattern in enumerate(stage.pattern.define()):
            
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
            setup_milling(microscope=microscope, milling_stage=stage)

            # draw pattern
            draw_pattern(microscope=microscope, pattern=pattern)

            # run milling
            run_milling(microscope=microscope, 
                        milling_current=stage.milling.milling_current, 
                        milling_voltage=stage.milling.milling_voltage, 
                        asynch=False)
            
            # finish milling (clear patterns, restore imaging current)
            finish_milling(
                microscope=microscope,
                imaging_current=microscope.system.ion.beam.beam_current,
                imaging_voltage=microscope.system.ion.beam.voltage,
            )

            # return to initial position
            microscope.move_stage_absolute(initial_position)