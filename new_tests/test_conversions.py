import pytest
from fibsem import conversions
from fibsem.structures import FibsemImage, Point
import numpy as np


@pytest.fixture
def points():

    point_o = Point(0,0)
    pointUR = Point(3,7)
    pointUL = Point(-5,8)
    pointDR = Point(8,-3)
    pointDL = Point(-4,-10)

    return [point_o,pointUR,pointUL,pointDR,pointDL]


def test_convert_m_to_p():

    distance_in_m = np.asarray([0.01,0.2,4,13,293])

    pixelsizes = np.asarray([1e-6,2.3e-6,2.354e-5,0.023,0.1])

    answers = distance_in_m/pixelsizes

    for i in range(len(answers)):

        output = conversions.convert_metres_to_pixels(distance_in_m[i],pixelsizes[i])

        assert output == int(answers[i]), f"output: {output} does not match answer: {int(answers)}"

def test_convert_p_to_m():

    pixels_num = np.random.randint(10000,size=10)

    pixel_sizes = np.random.rand(10)*1e-6

    answers = pixels_num*pixel_sizes

    for i in range(len(answers)):

        output = conversions.convert_pixels_to_metres(pixels_num[i],pixel_sizes[i])

        assert output == answers[i], f"output: {output} does not match answer: {answers[i]}"

def test_distance_between_points(points):

    p1 = points[0:4]
    p2 = points[1:5]

    answers = [0,0,0,0]

    for i in range(len(answers)):

        pointA = p1[i]
        pointB = p2[i]

        dist_x = pointB.x - pointA.x
        dist_y = pointB.y - pointA.y

        answers[i] = conversions.Point(x=dist_x,y=dist_y)

    for j in range(len(answers)):

        output = conversions.distance_between_points(p1[j],p2[j])

        assert output.x == answers[j].x, f"output x: {output.x} does not match answer {answers[j].x}"
        assert output.y == answers[j].y, f"output y: {output.y} does not match answer {answers[j].y}"

def test_convert_point_p_to_m(points):

    pixel_sizes = np.random.rand(len(points))*1e-6


    for i in range(len(points)):

        point = points[i]
        pxsize = pixel_sizes[i]

        x_check = point.x*pxsize
        y_check = point.y*pxsize

        output = conversions.convert_point_from_pixel_to_metres(point,pxsize)

        assert output.x == x_check, f"Output x: {output.x} does not match answer {x_check}"
        assert output.y == y_check, f"Output y: {output.y} does not match answer {y_check}"


def test_convert_point_m_to_p(points):

    pixel_sizes = np.random.rand(len(points))*1e-6


    for i in range(len(points)):

        point = points[i]
        pxsize = pixel_sizes[i]

        x_check = int(point.x/pxsize)
        y_check = int(point.y/pxsize)

        output = conversions.convert_point_from_metres_to_pixel(point,pxsize)

        assert output.x == x_check, f"Output x: {output.x} does not match answer {x_check}"
        assert output.y == y_check, f"Output y: {output.y} does not match answer {y_check}"



def test_image_to_microscope_image_coordinates(points,image):

    
    for i in range(len(points)):

        pointA = points[0]

        image_dim = np.random.randint(20,1000,2)

        image_array =np.random.randint(0,256,size=(image_dim[0],image_dim[1]))

        cy,cx = np.asarray(image_array.shape) // 2

        # dist from centre

        dy = -(pointA.y - cy)
        dx = pointA.x - cx



        point_m = conversions.image_to_microscope_image_coordinates(p1,image,0.001)

        assert point_m.x == 0
        assert point_m.y == 0

    

def test_get_lamella_size_in_pixels():

    pass


    

