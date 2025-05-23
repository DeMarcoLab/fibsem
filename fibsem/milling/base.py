from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import List, Union, Dict, Any, Tuple, Optional

from fibsem.microscope import FibsemMicroscope
from fibsem.milling.config import MILLING_SPUTTER_RATE
from fibsem.milling.patterning.patterns2 import BasePattern as BasePattern, get_pattern as get_pattern
from fibsem.structures import FibsemMillingSettings, Point, MillingAlignment, ImageSettings, CrossSectionPattern

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
    
    @property
    def advanced_attributes(self) -> List[str]:
        if hasattr(self, "_advanced_attributes"):
            return self._advanced_attributes
        return []

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
    def run(self, microscope: FibsemMicroscope, stage: "FibsemMillingStage", asynch: bool = False, parent_ui = None) -> None:
        pass


def get_strategy(
    name: str = "Standard", config: Optional[Dict[str, Any]] = None
) -> MillingStrategy:
    from fibsem.milling.strategy import get_strategies, DEFAULT_STRATEGY

    if config is None:
        config = {}

    strategies = get_strategies()
    return strategies.get(name, DEFAULT_STRATEGY).from_dict(config)


@dataclass
class FibsemMillingStage:
    name: str = "Milling Stage"
    num: int = 0
    milling: FibsemMillingSettings = FibsemMillingSettings()
    pattern: BasePattern = None
    patterns: List[BasePattern] = None # unused
    strategy: MillingStrategy = None
    alignment: MillingAlignment = MillingAlignment()
    imaging: ImageSettings = ImageSettings() # settings for post-milling acquisition

    def __post_init__(self):
        if self.pattern is None:
            self.pattern = get_pattern("Rectangle", 
                                       config={"width": 10e-6, "height": 5e-6, "depth": 1e-6})
        if self.strategy is None:
            self.strategy = get_strategy("Standard")

    def to_dict(self):
        return {
            "name": self.name,
            "num": self.num,
            "milling": self.milling.to_dict(),
            "pattern": self.pattern.to_dict(),
            "strategy": self.strategy.to_dict(),
            "alignment": self.alignment.to_dict(),
            "imaging": self.imaging.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict):
        strategy_config = data.get("strategy", {})
        strategy_name = strategy_config.get("name", "Standard")
        pattern_name = data["pattern"]["name"]
        alignment = data.get("alignment", {})
        imaging: dict = data.get("imaging", {})
        if imaging == {} or imaging.get("path", None) is None:
            imaging["path"] = None # set to None if not explicitly set
        return cls(
            name=data["name"],
            num=data.get("num", 0),
            milling=FibsemMillingSettings.from_dict(data["milling"]),
            pattern=get_pattern(pattern_name, data["pattern"]),
            strategy=get_strategy(strategy_name, config=strategy_config),
            alignment=MillingAlignment.from_dict(alignment),
            imaging=ImageSettings.from_dict(imaging)
        )

    @property
    def estimated_time(self) -> float:
        return estimate_milling_time(self.pattern, self.milling.milling_current)

    def run(self, microscope: FibsemMicroscope, asynch: bool = False, parent_ui = None) -> None:
        """Run the milling stage strategy on the given microscope."""
        self.strategy.run(microscope=microscope, stage=self, asynch=asynch, parent_ui=parent_ui)


def get_milling_stages(key: str, protocol: Dict[str, List[Dict[str, Any]]]) -> List[FibsemMillingStage]:
    """Get the milling stages for specific key from the protocol.
    Args:
        key: the key to get the milling stages for
        protocol: the protocol to get the milling stages from
    Returns:
        List[FibsemMillingStage]: the milling stages for the given key"""
    if key not in protocol:
        raise ValueError(f"Key {key} not found in protocol. Available keys: {list(protocol.keys())}")
    
    stages = []
    for stage_config in protocol[key]:
        stage = FibsemMillingStage.from_dict(stage_config)
        stages.append(stage)
    return stages

def get_protocol_from_stages(stages: List[FibsemMillingStage]) -> List[Dict[str, Any]]:
    """Convert a list of milling stages to a protocol dictionary.
    Args:
        stages: the list of milling stages to convert
    Returns:
        List[Dict[str, Any]]: the protocol dictionary"""
    if not isinstance(stages, list):
        stages = [stages]
    
    return deepcopy([stage.to_dict() for stage in stages])


def estimate_milling_time(pattern: BasePattern, milling_current: float) -> float:
    """Estimate the milling time for a given pattern and milling current. 
    The time is calculated as the volume of the pattern divided by the sputter rate at the given current.
    The sputter rate is taken from the microscope application files. 
    This is a rough estimate, as the actual milling time is calculated at milling time.

    Args:
        pattern (BasePattern): the milling pattern
        milling_current (float): the milling current in A

    Returns:
        float: the estimated milling time in seconds
    """
    # get the key that is closest to the milling current
    sp_keys = list(MILLING_SPUTTER_RATE.keys())
    sp_keys.sort(key=lambda x: abs(x - milling_current))

    # get the sputter rate for the closest key
    sputter_rate = MILLING_SPUTTER_RATE[sp_keys[0]] # um3/s 

    # scale the sputter rate based on the expected current
    sputter_rate = sputter_rate * (milling_current / sp_keys[0])
    volume = pattern.volume # m3

    if hasattr(pattern, "cross_section") and pattern.cross_section is CrossSectionPattern.CleaningCrossSection:
        volume *= 0.66 # ccs is approx 2/3 of the volume of a rectangle

    time = (volume *1e6**3) / sputter_rate
    return time * 0.75 # QUERY: accuracy of this estimate?

def estimate_total_milling_time(stages: List[FibsemMillingStage]) -> float:
    """Estimate the total milling time for a list of milling stages"""
    if not isinstance(stages, list):
        stages = [stages]
    return sum([estimate_milling_time(stage.pattern, stage.milling.milling_current) for stage in stages])
