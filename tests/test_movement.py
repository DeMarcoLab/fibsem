import pytest
import numpy as np

from fibsem import movement

def test_angle_difference():
    assert np.isclose(movement.angle_difference(np.deg2rad(0), np.deg2rad(0)), 0)
    assert np.isclose(movement.angle_difference(np.deg2rad(0), np.deg2rad(360)), 0)
    assert np.isclose(movement.angle_difference(np.deg2rad(0), np.deg2rad(180)), np.pi)
    assert np.isclose(movement.angle_difference(np.deg2rad(2), np.deg2rad(358)), np.deg2rad(4))
    assert np.isclose(movement.angle_difference(np.deg2rad(-45), np.deg2rad(45)), np.pi / 2)
    assert np.isclose(movement.angle_difference(np.deg2rad(-360), np.deg2rad(360)), 0)
    assert np.isclose(movement.angle_difference(-4*np.pi, 4*np.pi), 0)

def test_rotation_angle_is_larger():
    assert movement.rotation_angle_is_larger(np.deg2rad(0), np.deg2rad(0)) == False
    assert movement.rotation_angle_is_larger(np.deg2rad(0), np.deg2rad(360)) == False
    assert movement.rotation_angle_is_larger(np.deg2rad(-45), np.deg2rad(45)) == False
    assert movement.rotation_angle_is_larger(np.deg2rad(0), np.deg2rad(180)) == True
    assert movement.rotation_angle_is_larger(np.deg2rad(-90), np.deg2rad(90)) == True
    assert movement.rotation_angle_is_larger(np.deg2rad(0), np.deg2rad(720)) == False


def test_rotation_angle_is_smaller():
    assert movement.rotation_angle_is_smaller(np.deg2rad(0), np.deg2rad(0), 5) == True
    assert movement.rotation_angle_is_smaller(np.deg2rad(0), np.deg2rad(360), 5) == True
    assert movement.rotation_angle_is_smaller(np.deg2rad(0), np.deg2rad(180)) == False
    assert movement.rotation_angle_is_smaller(np.deg2rad(-90), np.deg2rad(90), 180) == False
    assert movement.rotation_angle_is_smaller(np.deg2rad(0), np.deg2rad(-360), 1) == True
