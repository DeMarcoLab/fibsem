from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
from fibsem.ui.FibsemMillingWidget import FibsemMillingWidget
from fibsem.ui.FibsemCryoDepositionWidget import FibsemCryoDepositionWidget
from fibsem.ui.FibsemMinimapWidget import FibsemMinimapWidget
from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget

try:
    from fibsem.ui.FibsemEmbeddedDetectionWidget import FibsemEmbeddedDetectionUI
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False
    import logging
    logging.debug("Could not import FibsemEmbeddedDetectionWidget")

