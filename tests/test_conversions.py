
import pytest

from fibsem import conversions
from fibsem.structures import Point

import numpy as np

def test_distance_between_points():

    point1 = Point(0, 0)
    point2 = Point(500, 500)

    diff = conversions.distance_between_points(point1, point2)

    assert diff.x == 500
    assert diff.y == 500

def test_convert_point_from_pixel_to_metres():

    point_px = Point(x=500, y=500)
    pixelsize = 1e-6

    point_m = conversions.convert_point_from_pixel_to_metres(point_px, pixelsize)

    assert point_m.x == point_px.x * pixelsize
    assert point_m.y == point_px.y * pixelsize


def test_convert_point_from_metres_to_pixel():

    point_m = Point(x=500e-6, y=500e-6)
    pixelsize = 1e-6

    point_px = conversions.convert_point_from_metres_to_pixel(point_m, pixelsize)

    assert point_px.x == int(point_m.x / pixelsize)
    assert point_px.y == int(point_m.y / pixelsize)

def test_convert_pixel_to_metres():
    
    pixels = 100
    pixelsize = 1e-6

    metres = conversions.convert_pixels_to_metres(pixels, pixelsize)

    assert np.isclose(metres, float(pixels * pixelsize))

def test_convert_metres_to_pixels():

    metres = 100e-6
    pixelsize = 1e-6

    pixels = conversions.convert_metres_to_pixels(metres, pixelsize)

    assert pixels == int(metres / pixelsize)

