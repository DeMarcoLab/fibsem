import logging
from dataclasses import dataclass

from fibsem.microscope import FibsemMicroscope
from fibsem.milling import (draw_patterns, run_milling,
                            setup_milling)
from fibsem.milling.base import (FibsemMillingStage, MillingStrategy,
                                 MillingStrategyConfig)

import time

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
    ) -> None:
        logging.info(f"Running {self.name} Milling Strategy for {stage.name}")
        setup_milling(microscope, milling_stage=stage)

        draw_patterns(microscope, stage.pattern.define())

        estimated_time = microscope.estimate_milling_time()
        logging.info(f"Estimated time for {stage.name}: {estimated_time:.2f} seconds")

        if parent_ui:
            parent_ui.milling_progress_signal.emit({"msg": f"Running {stage.name}...", 
                                                    "progress": 
                                                        {"started": True,
                                                            "start_time": time.time(), 
                                                            "estimated_time": estimated_time,
                                                            "name": stage.name}
                                                        })

        run_milling(
            microscope=microscope,
            milling_current=stage.milling.milling_current,
            milling_voltage=stage.milling.milling_voltage,
            asynch=asynch,
        )
