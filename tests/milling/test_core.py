import os
import glob
from pathlib import Path

from fibsem import utils
from fibsem.milling import FibsemMillingStage, mill_stages


def test_mill_stages_with_acquisitions(tmp_path: Path) -> None:
    """Test milling stages with image acquisitions (alignment, reference images, etc.).
    Smoke test to ensure images are being acquired and saved correctly.
    """
    # setup a microscope session
    microscope, settings = utils.setup_session(manufacturer="Demo")

    milling_stage = FibsemMillingStage(name="test-stage")
    milling_stage.milling.acquire_images = True
    milling_stage.alignment.enabled = True
    milling_stage.imaging.path = tmp_path
    milling_stage.imaging.save = True
    mill_stages(microscope, [milling_stage])

    # check for the expected reference image
    glob_pattern = f"ref_{milling_stage.name}_initial_alignment*.tif"
    files = glob.glob(os.path.join(tmp_path, glob_pattern))
    assert len(files) > 0, f"No files found in {tmp_path} with pattern {glob_pattern}"

    # check for the beam shift alignment files
    glob_pattern = "beam_shift_alignment*.tif"
    files = glob.glob(os.path.join(tmp_path, glob_pattern))
    assert len(files) > 0, f"No files found in {tmp_path} with pattern {glob_pattern}"

    # check for the post acquisition reference image
    glob_pattern = f"ref_milling_{milling_stage.name}_finished_*.tif"
    files = glob.glob(os.path.join(tmp_path, glob_pattern))
    assert len(files) == 2, f"No files found in {tmp_path} with pattern {glob_pattern}"
