import pytest
from fibsem import structures

def test_point():

    p1 = structures.Point(2,3)

    p_dict = p1.__to_dict__()

    assert p_dict["x"] == 2
    assert p_dict["y"] == 3

    p_made_dict = {"x":5,"y":6}

    p2 = structures.Point.__from_dict__(p_made_dict)

    assert p2.x == 5
    assert p2.y == 6

    p_list = p2.__to_list__()

    assert p_list[0] == 5
    assert p_list[1] == 6

    p_made_list = [-2,-9]

    p3 = structures.Point.__from_list__(p_made_list)

    assert p3.x == -2
    assert p3.y == -9

    p1.x = 5

    assert p1.x == 5

    with pytest.raises(Exception):

        bad_point = structures.Point("a","b")



