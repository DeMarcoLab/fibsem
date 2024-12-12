import logging
from dataclasses import dataclass

from fibsem.microscope import FibsemMicroscope
from fibsem.milling import (draw_patterns, estimate_milling_time, run_milling,
                            setup_milling)
from fibsem.milling.base import (FibsemMillingStage, MillingStrategy,
                                 MillingStrategyConfig)


@dataclass
class StandardMillingConfig(MillingStrategyConfig):
    """Configuration for standard milling strategy"""
    pass


@dataclass
class StandardMillingStrategy(MillingStrategy):
    """Basic milling strategy that mills continuously until completion"""
    name: str = "Standard"
    fullname: str = "Standard Milling"

    def __init__(self, config: StandardMillingConfig = None):
        self.config = config or StandardMillingConfig()

    def to_dict(self):
        return {"name": self.name, "config": self.config.to_dict()}

    @staticmethod
    def from_dict(d: dict) -> "StandardMillingStrategy":
        config=StandardMillingConfig.from_dict(d.get("config", {}))   
        return StandardMillingStrategy(config=config)

    def run(
        self,
        microscope: FibsemMicroscope,
        stage: FibsemMillingStage,
        asynch: bool = False,
        parent_ui = None,
        current_stage_index: int = 0,
        total_stages: int = 1,
    ) -> None:
        logging.info(f"Running {self.name} Milling Strategy for {stage.name}")
        setup_milling(microscope, milling_stage=stage)

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
