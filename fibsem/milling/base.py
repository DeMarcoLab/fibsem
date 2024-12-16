from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import List, Union, Dict, Any, Tuple

from fibsem.microscope import FibsemMicroscope
from fibsem.milling.patterning.patterns2 import BasePattern as BasePattern, get_pattern as get_pattern
from fibsem.structures import FibsemMillingSettings, Point, MillingDriftCorrection

@dataclass
class MillingStrategyConfig(ABC):
    """Abstract base class for milling strategy configurations"""
    
    def to_dict(self):
        return {}

    @staticmethod
    def from_dict(d: dict) -> "MillingStrategyConfig":
        return MillingStrategyConfig()
    
    @property
    def required_attributes(self) -> Tuple[str]:
        return [field.name for field in fields(self)]

@dataclass
class MillingStrategy(ABC):
    """Abstract base class for different milling strategies"""
    name: str = "Milling Strategy"    
    config = MillingStrategyConfig()

    def __init__(self, **kwargs):
        pass
    
    @abstractmethod
    def to_dict(self):
        return {"name": self.name, "config": self.config.to_dict()}
    
    @staticmethod
    @abstractmethod
    def from_dict(d: dict) -> "MillingStrategy":
        pass

    @abstractmethod
    def run(self, microscope: FibsemMicroscope, stage: "FibsemMillingStage", asynch: bool = False) -> None:
        pass

def get_strategy(name: str = "Standard", config: Dict[str, Any] = {}) -> MillingStrategy:
    from fibsem.milling.strategy import strategies, DEFAULT_STRATEGY
    return strategies.get(name, DEFAULT_STRATEGY).from_dict(config)


@dataclass
class FibsemMillingStage:
    name: str = "Milling Stage"
    num: int = 0
    milling: FibsemMillingSettings = FibsemMillingSettings()
    pattern: BasePattern = None
    patterns: List[BasePattern] = None # unused
    strategy: MillingStrategy = None
    drift_correction: MillingDriftCorrection = MillingDriftCorrection()

    def __post_init__(self):
        if self.pattern is None:
            self.pattern = get_pattern("Rectangle")
        if self.strategy is None:
            self.strategy = get_strategy("Standard")

    def to_dict(self):
        return {
            "name": self.name,
            "num": self.num,
            "milling": self.milling.to_dict(),
            "pattern": self.pattern.to_dict(),
            "strategy": self.strategy.to_dict(),
            "drift_correction": self.drift_correction.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict):
        strategy_config = data.get("strategy", {})
        strategy_name = strategy_config.get("name", "Standard")
        pattern_name = data["pattern"]["name"]
        drift_correction = data.get("drift_correction", {})
        return cls(
            name=data["name"],
            num=data.get("num", 0),
            milling=FibsemMillingSettings.from_dict(data["milling"]),
            pattern=get_pattern(pattern_name, data["pattern"]),
            strategy=get_strategy(strategy_name, config=strategy_config),
            drift_correction=MillingDriftCorrection.from_dict(drift_correction),
        )

def get_milling_stages(key: str, protocol: Dict[str, List[Dict[str, Any]]]) -> List[FibsemMillingStage]:
    stages = []
    for stage_config in protocol[key]:
        stage = FibsemMillingStage.from_dict(stage_config)
        stages.append(stage)
    return stages


def get_protocol_from_stages(stages: List[FibsemMillingStage]) -> List[Dict[str, Any]]:
    
    if not isinstance(stages, list):
        stages = [stages]
    
    return deepcopy([stage.to_dict() for stage in stages])