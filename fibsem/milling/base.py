from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, fields, field, asdict
from typing import List, Union, Dict, Any, Tuple, Optional, Type, TypeVar, ClassVar, Generic

from fibsem.microscope import FibsemMicroscope
from fibsem.milling.config import MILLING_SPUTTER_RATE
from fibsem.milling.patterning.patterns2 import BasePattern
from fibsem.milling.patterning import get_pattern, DEFAULT_MILLING_PATTERN
from fibsem.structures import (
    FibsemMillingSettings,
    MillingAlignment,
    ImageSettings,
    CrossSectionPattern,
    FibsemImage,
)


TMillingStrategyConfig = TypeVar(
    "TMillingStrategyConfig", bound="MillingStrategyConfig"
)
TMillingStrategy = TypeVar("TMillingStrategy", bound="MillingStrategy")


@dataclass
class MillingStrategyConfig(ABC):
    """Abstract base class for milling strategy configurations"""
    _advanced_attributes: ClassVar[Tuple[str, ...]] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls: Type[TMillingStrategyConfig], d: Dict[str, Any]
    ) -> TMillingStrategyConfig:
        return cls(**d)

    @property
    def required_attributes(self) -> Tuple[str, ...]:
        return tuple(f.name for f in fields(self))

    @property
    def advanced_attributes(self) -> Tuple[str, ...]:
        """Attributes that are considered advanced and may not be required for all strategies."""
        return self._advanced_attributes


class MillingStrategy(ABC, Generic[TMillingStrategyConfig]):
    """Abstract base class for different milling strategies"""
    name: str = "Milling Strategy"
    config_class: Type[TMillingStrategyConfig]

    def __init__(self, config: Optional[TMillingStrategyConfig] = None) -> None:
        self.config: TMillingStrategyConfig = config or self.config_class()

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "config": self.config.to_dict()}

    @classmethod
    def from_dict(cls: Type[TMillingStrategy], d: Dict[str, Any]) -> TMillingStrategy:
        config = cls.config_class.from_dict(d.get("config", {}))
        return cls(config=config)

    @abstractmethod
    def run(self, microscope: FibsemMicroscope, stage: "FibsemMillingStage", asynch: bool = False, parent_ui = None) -> None:
        pass


def get_strategy(
    name: str = "Standard", config: Optional[Dict[str, Any]] = None
) -> MillingStrategy[Any]:
    from fibsem.milling.strategy import get_strategies, DEFAULT_STRATEGY

    if config is None:
        config = {}

    strategies = get_strategies()
    return strategies.get(name, DEFAULT_STRATEGY).from_dict(config)


@dataclass
class FibsemMillingStage:
    name: str = "Milling Stage"
    num: int = 0
    milling: FibsemMillingSettings = field(default_factory=FibsemMillingSettings)
    pattern: BasePattern = field(default_factory=DEFAULT_MILLING_PATTERN)
    patterns: Optional[List[BasePattern]] = None # unused
    strategy: MillingStrategy[Any] = field(default_factory=get_strategy)
    alignment: MillingAlignment = field(default_factory=MillingAlignment)
    imaging: ImageSettings = field(default_factory=ImageSettings) # settings for post-milling acquisition
    reference_image: Optional[FibsemImage] = None

    def __post_init__(self):
        
        if self.imaging.resolution is None:
            self.imaging.resolution = [1536, 1024]  # default resolution for imaging
        if self.imaging.hfw is None:
            self.imaging.hfw = 150e-6
        if self.imaging.dwell_time is None:
            self.imaging.dwell_time = 1e-6
        if self.imaging.autocontrast is None:
            self.imaging.autocontrast = False
        if self.imaging.save is None:
            self.imaging.save = False

    def to_dict(self):
        return {
            "name": self.name,
            "num": self.num,
            "milling": self.milling.to_dict(),
            "pattern": self.pattern.to_dict(),
            "strategy": self.strategy.to_dict(),
            "alignment": self.alignment.to_dict(),
            "imaging": self.imaging.to_dict(),
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
            imaging=ImageSettings.from_dict(imaging),
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

def get_protocol_from_stages(stages: Union[FibsemMillingStage, List[FibsemMillingStage]]) -> List[Dict[str, Any]]:
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


