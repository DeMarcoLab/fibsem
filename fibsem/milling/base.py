from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Union

from fibsem.microscope import FibsemMicroscope
from fibsem.milling.patterning.patterns import BasePattern, _get_pattern, get_pattern
from fibsem.structures import FibsemMillingSettings, Point, MillingDriftCorrection

@dataclass
class MillingStrategyConfig(ABC):
    """Abstract base class for milling strategy configurations"""
    
    def to_dict(self):
        return {}

    @staticmethod
    def from_dict(d: dict) -> "MillingStrategyConfig":
        return MillingStrategyConfig()

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

def get_strategy(name: str = "Standard", config: dict = {}) -> MillingStrategy:
    from fibsem.milling.strategy import strategies, DEFAULT_STRATEGY
    return strategies.get(name, DEFAULT_STRATEGY).from_dict(config)


@dataclass
class FibsemMillingStage:
    name: str = "Milling Stage"
    num: int = 0
    milling: FibsemMillingSettings = FibsemMillingSettings()
    pattern: BasePattern = None # TODO: list
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
        }

    @classmethod
    def from_dict(cls, data: dict):
        strategy_config = data.get("strategy", {})
        strategy_name = strategy_config.get("name", "Standard")
        return cls(
            name=data["name"],
            num=data["num"],
            milling=FibsemMillingSettings.from_dict(data["milling"]),
            pattern=get_pattern(data["pattern"]["name"]).from_dict(data["pattern"]),
            strategy=get_strategy(strategy_name, config=strategy_config),
        )

def _get_stage(key, protocol: dict, point: Point = Point(), i: int = 0) -> FibsemMillingStage:
    pattern = _get_pattern(key, protocol, point=point)
    mill_settings = FibsemMillingSettings(
        milling_current=protocol["milling_current"], 
        hfw=float(protocol["hfw"]),
        application_file=protocol.get("application_file", "Si"),
        preset=protocol.get("preset", "30 keV; 20 nA"),
        patterning_mode=protocol.get("patterning_mode", "Serial"))

    # milling stage name
    name = protocol.get("name", f"{key.title()} {i+1:02d}")

    # strategy
    strategy_config = protocol.get("strategy", {})
    strategy_name = strategy_config.get("name", "Standard")
    strategy = get_strategy(strategy_name, config=strategy_config)

    stage = FibsemMillingStage(
        name=name, num=i, milling=mill_settings, pattern=pattern, strategy=strategy
    )
    # TODO: migrate to use from_dict?
    return stage

def get_milling_stages(key, protocol, point: Union[Point, List[Point]] = Point()):
    
    # TODO: maybe add support for defining point per stages?

    # convert point to list of points, same length as stages

    if "stages" in protocol[key]:
        stages = []
        for i, pstage in enumerate(protocol[key]["stages"]):
            
            if isinstance(point, list):
                pt = point[i]
            else:
                pt = point

            stages.append(_get_stage(key, pstage, point=pt, i=i))
    else:
        stages = [_get_stage(key, protocol[key], point=point)]
    return stages

def get_protocol_from_stages(stages: list):
    protocol = {}  
    protocol["stages"] = []

    if not isinstance(stages, list):
        stages = [stages]

    for stage in stages:
        # join two dicts union
        ddict = {**stage.to_dict()["milling"], 
            **stage.to_dict()["pattern"]["protocol"], 
            "type": stage.to_dict()["pattern"]["name"],
            "name": stage.name, 
            "strategy": stage.strategy.to_dict(),}
        protocol["stages"].append(deepcopy(ddict))

    return protocol