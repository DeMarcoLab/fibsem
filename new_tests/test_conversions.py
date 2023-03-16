import pytest
from fibsem import conversions
from fibsem.structures import FibsemImage, Point


@pytest.fixture
def points():

    point1 = Point(1,2)
    point2 = Point(-3,7)
    point3 = Point(0,0)
    point4 = Point(8,-3)
    point5 = Point(-4,-10)

    return [point1,point2,point3,point4,point5]


def test_image_to_microscope_image_coordinates():
    pass

def test_get_lamella_size_in_pixels():

    pass

def test_convert_meters_to_pixels():

    pass

def test_convert_pixels_to_meters():

    pass

def test_distance_between_points():

    pass

def test_convert_point_p_to_m():

    pass

def test_convert_point_m_to_p():

    pass
    

