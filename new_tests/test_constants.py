import pytest
from fibsem import constants
import numpy as np

@pytest.fixture
def values():

    values = [23, 1.2, 3.3285847, -37.2, 0.000045, 1092334]

    return values


def test_meter_micron(values):

    for value in values:

        v1 = value*constants.MICRON_TO_METRE
        c1 = value*1e-6

        v2 = value*constants.METRE_TO_MICRON
        c2 = value*1e6

        assert v1 == c1, f"{v1} is not equivalent to {c1}"
        assert v2 == c2, f"{v2} is not equivalent to {c2}"

def test_meter_mm(values):

    for value in values:

        v1 = value*constants.MILLIMETRE_TO_METRE
        c1 = value*1e-3

        v2 = value*constants.METRE_TO_MILLIMETRE
        c2 = value*1e3

        assert v1 == c1, f"{v1} is not equivalent to {c1}"
        assert v2 == c2, f"{v2} is not equivalent to {c2}"
    

def test_SI_milli(values):

    for value in values:

        v1 = value*constants.SI_TO_MILLI
        c1 = value*1e3

        v2 = value*constants.MILLI_TO_SI
        c2 = value*1e-3

        assert v1 == c1, f"{v1} is not equivalent to {c1}"
        assert v2 == c2, f"{v2} is not equivalent to {c2}"

def test_SI_KILO(values):

       for value in values:

        v1 = value*constants.SI_TO_KILO
        c1 = value*1e-3

        v2 = value*constants.KILO_TO_SI
        c2 = value*1e3

        assert v1 == c1, f"{v1} is not equivalent to {c1}"
        assert v2 == c2, f"{v2} is not equivalent to {c2}"

def test_SI_Micro(values):

    for value in values:

        v1 = value*constants.MICRO_TO_SI
        c1 = value*1e-6

        v2 = value*constants.SI_TO_MICRO
        c2 = value*1e6

        assert v1 == c1, f"{v1} is not equivalent to {c1}"
        assert v2 == c2, f"{v2} is not equivalent to {c2}"


def test_SI_Nano(values):

    for value in values:

        v1 = value*constants.NANO_TO_SI
        c1 = value*1e-9

        v2 = value*constants.SI_TO_NANO
        c2 = value*1e9

        assert v1 == c1, f"{v1} is not equivalent to {c1}"
        assert v2 == c2, f"{v2} is not equivalent to {c2}"
    


def test_SI_Pico(values):

    for value in values:

        v1 = value*constants.PICO_TO_SI
        c1 = value*1e-12

        v2 = value*constants.SI_TO_PICO
        c2 = value*1e12

        assert v1 == c1, f"{v1} is not equivalent to {c1}"
        assert v2 == c2, f"{v2} is not equivalent to {c2}"


def test_angle_conversion():

    # angle conversions are good to 5 decimal places, accuracy loss in further precision

    angles = [20,5,-1,0.3,0.002,33.56]

    for angle in angles:

        v1 = angle*constants.RADIANS_TO_DEGREES
        c1 = angle*180/np.pi

        v2 = angle*constants.DEGREES_TO_RADIANS
        c2 = angle*np.pi/180

        assert round(v1,5) == round(c1,5), f"{v1} is not equivalent to {c1}"
        assert round(v2,5) == round(c2,5), f"{v2} is not equivalent to {c2}"
