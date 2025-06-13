from fibsem.milling.base import (
    FibsemMillingStage,
    MillingStrategy,
    get_strategy,
)
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.milling.strategy import DEFAULT_STRATEGY, get_strategies
from fibsem.structures import FibsemMillingSettings, MillingAlignment, Point

def test_milling_stage():

    milling_settings = FibsemMillingSettings()
    pattern = RectanglePattern(width=10, height=5, depth=1)
    strategy = get_strategy("Standard")
    alignment = MillingAlignment(enabled=True)

    # Create a FibsemMillingStage instance
    milling_stage = FibsemMillingStage(
        name="Test Stage",
        num=1,
        milling=milling_settings,
        pattern=pattern,
        strategy=strategy,
        alignment=alignment,
    )

    # Check the attributes
    assert milling_stage.name == "Test Stage"
    assert milling_stage.num == 1
    assert isinstance(milling_stage.milling, FibsemMillingSettings)
    assert isinstance(milling_stage.pattern, RectanglePattern)
    assert isinstance(milling_stage.strategy, MillingStrategy)
    assert isinstance(milling_stage.alignment, MillingAlignment)
    assert milling_stage.pattern.width == 10
    assert milling_stage.pattern.height == 5
    assert milling_stage.pattern.depth == 1
    assert milling_stage.strategy.name == DEFAULT_STRATEGY.name
    assert milling_stage.alignment.enabled is True

    # test to_dict method
    dict_repr = milling_stage.to_dict()
    assert dict_repr["name"] == "Test Stage"
    assert dict_repr["num"] == 1
    assert isinstance(dict_repr["milling"], dict)
    assert isinstance(dict_repr["pattern"], dict)
    assert isinstance(dict_repr["strategy"], dict)
    assert isinstance(dict_repr["alignment"], dict)

    # test from_dict method
    dict_repr = {
        "name": "Test Stage",
        "num": 1,
        "milling": milling_settings.to_dict(),
        "pattern": pattern.to_dict(),
        "strategy": strategy.to_dict(),
        "alignment": alignment.to_dict(),
    }
    new_milling_stage = FibsemMillingStage.from_dict(dict_repr)


    assert new_milling_stage.name == "Test Stage"
    assert new_milling_stage.num == 1
    assert isinstance(new_milling_stage.milling, FibsemMillingSettings)
    assert isinstance(new_milling_stage.pattern, RectanglePattern)
    assert isinstance(new_milling_stage.strategy, MillingStrategy)
    assert isinstance(new_milling_stage.alignment, MillingAlignment)
    assert new_milling_stage.imaging.path is None

def test_get_strategy():
    # Test with default strategy
    strategy = get_strategy()
    assert isinstance(strategy, MillingStrategy)
    assert strategy.name == DEFAULT_STRATEGY.name

    # Test with a specific strategy name
    strategy_name = "Standard"
    strategy = get_strategy(name=strategy_name)
    assert isinstance(strategy, MillingStrategy)
    assert strategy.name == strategy_name

    # Test with a non-existent strategy name
    strategy = get_strategy(name="NonExistentStrategy")
    assert isinstance(strategy, MillingStrategy)
    assert strategy.name == DEFAULT_STRATEGY.name
