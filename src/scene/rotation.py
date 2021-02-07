import numpy as np


def rotation_x(phi: float) -> np.ndarray:
    """
    Return the Rotation matrix around the X-axis for phi radians.
    """
    R_rot = np.array([[1, 0, 0],
                      [0, np.cos(phi), -np.sin(phi)],
                      [0, np.sin(phi), np.cos(phi)]])
    return R_rot


def rotation_y(theta: float) -> np.ndarray:
    """
    Return the Rotation matrix around the Y-axis for theta radians.
    """
    R_rot = np.array([[np.cos(theta), 0, -np.sin(theta)],
                      [0, 1, 0],
                      [np.sin(theta), 0, np.cos(theta)]])
    return R_rot


def rotation_z(psi: float) -> np.ndarray:
    """
    Return the Rotation matrix around the Z-axis for psi radians.
    """
    R_rot = np.array([[np.cos(psi), -np.sin(psi), 0],
                      [np.sin(psi), np.cos(psi), 0],
                      [0, 0, 1]])
    return R_rot


def euler_rotation(roll: float, pitch: float, yaw: float, is_radians: bool = False) -> np.ndarray:
    """
    Returns the Euler rotation matrix for 3 rotation angles around X-Y-Z
    """
    # Convert to Radians:
    if not is_radians:
        roll, pitch, yaw = np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)

    R = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)
    return R
