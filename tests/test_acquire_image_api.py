import pytest

from fibsem import utils
from fibsem.structures import MicroscopeSettings, BeamType, ImageSettings, FibsemImage
from fibsem.microscope import FibsemMicroscope

@pytest.fixture
def demo_microscope() -> tuple[FibsemMicroscope, MicroscopeSettings]:
    """Fixture that provides a demo microscope and settings for testing."""
    microscope, settings = utils.setup_session(manufacturer="Demo")
    return microscope, settings


def test_parameter_validation(demo_microscope):
    """Test parameter validation for the new acquire_image API."""
    microscope, settings = demo_microscope
    
    # Should raise ValueError when neither parameter provided
    with pytest.raises(ValueError, match="Must provide either image_settings.*or beam_type"):
        microscope.acquire_image()
    
    # Should work with only image_settings
    image = microscope.acquire_image(image_settings=settings.image)
    assert isinstance(image, FibsemImage)
    assert image is not None
    
    # Should work with only beam_type
    image = microscope.acquire_image(beam_type=BeamType.ELECTRON)
    assert isinstance(image, FibsemImage)
    assert image is not None
    
    # Should work with both parameters (image_settings takes precedence)
    image = microscope.acquire_image(image_settings=settings.image, beam_type=BeamType.ION)
    assert isinstance(image, FibsemImage)
    assert image is not None


def test_image_settings_precedence(demo_microscope):
    """Test that image_settings takes precedence when both parameters are provided."""
    microscope, settings = demo_microscope
    
    # Create specific settings with ELECTRON beam
    electron_settings = ImageSettings(
        beam_type=BeamType.ELECTRON, 
        hfw=1e-6, 
        resolution=(1024, 1024),
        dwell_time=1e-6
    )
    
    # Provide both image_settings (ELECTRON) and beam_type (ION)
    # image_settings should take precedence
    image = microscope.acquire_image(image_settings=electron_settings, beam_type=BeamType.ION)
    
    # Verify image_settings took precedence over beam_type parameter
    assert image.metadata.image_settings.beam_type == BeamType.ELECTRON
    assert image.metadata.image_settings.hfw == 1e-6
    assert image.metadata.image_settings.resolution == (1024, 1024)
    assert image.metadata.image_settings.dwell_time == 1e-6


def test_current_settings_usage(demo_microscope):
    """Test that beam_type parameter uses current microscope settings."""
    microscope, settings = demo_microscope
    
    # Set up some specific current settings for ELECTRON beam
    test_hfw = 50e-6
    microscope.set_field_of_view(hfw=test_hfw, beam_type=BeamType.ELECTRON)
    
    # Acquire with beam_type only - should use current microscope settings
    image: FibsemImage = microscope.acquire_image(beam_type=BeamType.ELECTRON)
    
    # Get current settings to compare
    current_settings = microscope.get_imaging_settings(beam_type=BeamType.ELECTRON)
    
    # Should use current microscope settings
    assert image.metadata.image_settings.beam_type == BeamType.ELECTRON
    assert image.metadata.image_settings.hfw == current_settings.hfw
    assert abs(image.metadata.image_settings.hfw - test_hfw) < 1e-9  # Should be close to what we set
    
    # Test with ION beam as well
    test_hfw_ion = 100e-6
    microscope.set_field_of_view(hfw=test_hfw_ion, beam_type=BeamType.ION)
    
    image_ion: FibsemImage = microscope.acquire_image(beam_type=BeamType.ION)
    current_settings_ion = microscope.get_imaging_settings(beam_type=BeamType.ION)
    
    assert image_ion.metadata.image_settings.beam_type == BeamType.ION
    assert image_ion.metadata.image_settings.hfw == current_settings_ion.hfw
    assert abs(image_ion.metadata.image_settings.hfw - test_hfw_ion) < 1e-9

def test_metadata_consistency(demo_microscope):
    """Test that metadata is correctly set for both calling patterns."""
    microscope, settings = demo_microscope
    
    # Test with image_settings
    image1 = microscope.acquire_image(image_settings=settings.image)
    assert image1.metadata.user == microscope.user
    assert image1.metadata.experiment == microscope.experiment
    assert image1.metadata.system == microscope.system
    assert image1.metadata.image_settings is not None
    assert image1.metadata.microscope_state is not None
    
    # Test with beam_type only
    image2 = microscope.acquire_image(beam_type=BeamType.ELECTRON)
    assert image2.metadata.user == microscope.user
    assert image2.metadata.experiment == microscope.experiment
    assert image2.metadata.system == microscope.system
    assert image2.metadata.image_settings is not None
    assert image2.metadata.microscope_state is not None
    
    # Both should have valid metadata structure
    assert hasattr(image1.metadata, 'image_settings')
    assert hasattr(image2.metadata, 'image_settings')