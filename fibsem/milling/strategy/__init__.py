from typing import Dict
from fibsem.milling.base import MillingStrategy
from fibsem.milling.strategy.standard import StandardMillingStrategy
from fibsem.milling.strategy.overtilt import OvertiltTrenchMillingStrategy

DEFAULT_STRATEGY = StandardMillingStrategy.name
strategies: Dict[str, MillingStrategy] = {
    StandardMillingStrategy.name: StandardMillingStrategy,
    OvertiltTrenchMillingStrategy.name: OvertiltTrenchMillingStrategy,
}
MILLING_STRATEGY_NAMES = list(strategies.keys())

def register_strategy(strategy_cls: MillingStrategy):
    strategies[strategy_cls.name] = strategy_cls
    MILLING_STRATEGY_NAMES.append(strategy_cls.name)
