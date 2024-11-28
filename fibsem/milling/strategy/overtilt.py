
from dataclasses import dataclass
import numpy as np

from fibsem.microscope import FibsemMicroscope
from fibsem.milling import draw_pattern, run_milling, setup_milling
from fibsem.milling.base import FibsemMillingStage, MillingStrategy
from fibsem.milling.patterning import patterns
from fibsem.structures import BeamType, FibsemImage, ImageSettings, FibsemStagePosition


@dataclass
class OvertiltTrenchMillingStrategy(MillingStrategy):
    """Overtilt milling strategy for trench milling"""
    name: str = "Overtilt Milling Strategy"

    def __init__(self):
        pass

    def run(self, microscope: FibsemMicroscope, stage: "FibsemMillingStage", asynch: bool = False) -> None:

        """Mill a trench pattern with overtilt, 
        based on https://www.sciencedirect.com/science/article/abs/pii/S1047847716301514 and autolamella v1"""
        # set up milling
        setup_milling(microscope, stage.milling)

        from fibsem import alignment

        overtilt_in_degrees = stage.pattern.protocol.get("overtilt", 1)

        # assert pattern is TrenchPattern
        if not isinstance(stage.pattern, patterns.TrenchPattern):
            raise ValueError("Pattern must be TrenchPattern for overtilt milling")

        # save initial position
        initial_position = microscope.get_stage_position()

        # TODO: we should acquire the initial reference image here, 
        # just pass in params

        for i, pattern in enumerate(stage.pattern.patterns):
            
            # overtilt
            if i == 0:
                t = -np.deg2rad(overtilt_in_degrees)
            else:
                t = +np.deg2rad(overtilt_in_degrees)
            microscope.move_stage_relative(FibsemStagePosition(t=t))

            # beam alignment
            image_settings = ImageSettings.fromFibsemImage(self.ref_image)
            image_settings.filename = f"alignment_target_{stage.name}"
            
            alignment.multi_step_alignment_v2(microscope=microscope, 
                                            ref_image=self.ref_image, 
                                            beam_type=BeamType.ION, 
                                            alignment_current=None,
                                            steps=3)

            # setup again to ensure we are milling at the correct current, cleared patterns
            setup_milling(microscope, stage.milling)

            # draw pattern
            draw_pattern(microscope, pattern)

            # run milling
            run_milling(microscope, stage.milling.milling_current, asynch)

            # return to initial position
            microscope.move_stage_absolute(initial_position)
