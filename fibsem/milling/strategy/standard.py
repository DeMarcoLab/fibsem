import logging
from dataclasses import dataclass

from fibsem.microscope import FibsemMicroscope
from fibsem.milling.base import FibsemMillingStage, MillingStrategy
from fibsem.milling import (
    draw_patterns,
    run_milling,
    setup_milling,
    estimate_milling_time,
)


@dataclass
class StandardMillingStrategy(MillingStrategy):
    """Basic milling strategy that mills continuously until completion"""

    name: str = "Standard"

    def run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        asynch: bool = False,
        parent_ui = None,
        current_stage_index: int = 0,
        total_stages: int = 1,
    ) -> None:
        logging.info(f"Running {self.name} for {stage.name}")
        setup_milling(microscope, stage.milling)

        patterns = draw_patterns(microscope, stage.pattern.patterns)

        estimated_time = estimate_milling_time(microscope, patterns)
        logging.info(f"Estimated time for {stage.name}: {estimated_time:.2f} seconds")

        if parent_ui:
            progress_bar_dict = {
                "estimated_time": estimated_time,
                "idx": current_stage_index,
                "total": total_stages,
            }
            parent_ui._progress_bar_start.emit(progress_bar_dict)
            parent_ui.milling_notification.emit(f"Running {stage.name}...")

        run_milling(
            microscope=microscope,
            milling_current=stage.milling.milling_current,
            milling_voltage=stage.milling.milling_voltage,
            asynch=asynch,
        )
