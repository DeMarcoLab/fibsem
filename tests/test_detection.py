import pytest


# import numpy as np
# from fibsem.structures import Point
# from fibsem.detection import detection, utils




# def test_detect_corner():

#     # set square pattern
#     arr = np.ones(shape=(100, 100))
#     arr = np.pad(arr, pad_width=100, mode = "constant", constant_values=0)

#     # detect corners
#     top_right = detection.detect_corner(arr)
#     top_left = detection.detect_corner(arr, left=True)
#     bottom_right = detection.detect_corner(arr, left=False, bottom=True)
#     bottom_left = detection.detect_corner(arr, left=True, bottom=True)

#     assert top_right == Point(199, 100)
#     assert top_left == Point(100, 100)
#     assert bottom_right == Point(199, 199)
#     assert bottom_left == Point(100, 199)


# def test_detect_closest_edge():

#     # set square pattern
#     arr = np.ones(shape=(100, 100))
#     arr = np.pad(arr, pad_width=100, mode = "constant", constant_values=0)

#     # detect corners 
#     pt = Point(0, 0)
#     top = detection.detect_closest_edge_v2(arr, pt)

#     pt = Point(299, 299)
#     bot = detection.detect_closest_edge_v2(arr, pt)

#     assert top == Point(100, 100)
#     assert bot == Point(199, 199)

