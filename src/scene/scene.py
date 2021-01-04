from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List
from .colors import *
from .common import Vector3D, Color


@dataclass
class Sphere:
    origin: Vector3D
    radius: float
    color: Color

    data_length: int = 7

    def to_array(self) -> np.ndarray:
        data = np.zeros(self.data_length, dtype=np.float32)
        data[0:3] = np.array(self.origin)
        data[3] = self.radius
        data[4:7] = np.array(self.color)

        return data


@dataclass
class Light:
    origin: Vector3D

    data_length: int = 3

    def to_array(self) -> np.ndarray:
        data = np.zeros(self.data_length, dtype=np.float32)
        data[0:3] = np.array(self.origin)

        return data


@dataclass
class Plane:
    origin: Vector3D
    normal: Vector3D
    color: Color

    data_length: int = 9

    def to_array(self) -> np.ndarray:
        data = np.zeros(self.data_length, dtype=np.float32)
        data[0:3] = np.array(self.origin)
        data[3:6] = np.array(self.normal) / np.linalg.norm(np.array(self.normal))
        data[6:9] = np.array(self.color)

        return data


class Scene:
    def __init__(self, lights: List[Light], spheres: List[Sphere], planes: List[Plane]):
        self.lights = lights
        self.spheres = spheres
        self.planes = planes

    def get_spheres(self) -> np.ndarray:
        """ Generate data array containing sphere data: """
        data = np.zeros((Sphere.data_length, len(self.spheres)), dtype=np.float32)

        for i, s in enumerate(self.spheres):
            data[:, i] = s.to_array()

        return data

    def get_planes(self) -> np.ndarray:
        """ Generate data array containing plane data: """
        data = np.zeros((Plane.data_length, len(self.planes)), dtype=np.float32)

        for i, p in enumerate(self.planes):
            data[:, i] = p.to_array()

        return data

    def get_lights(self) -> np.ndarray:
        """ Generate data array containing sphere data: """
        data = np.zeros((Light.data_length, len(self.lights)), dtype=np.float32)

        for i, l in enumerate(self.lights):
            data[:, i] = l.to_array()

        return data

    def generate_scene(self) -> (np.ndarray, np.ndarray, np.ndarray):
        return self.get_spheres(), self.get_lights(), self.get_planes()

    @staticmethod
    def default_scene() -> Scene:

        lights = [Light([2.5, -2.0, 3.0]),
                  Light([2.5,  2.0, 3.0])]

        spheres = [Sphere([2.2, 0.3, 1.0], 1.0, RED),
                   Sphere([0.6, 0.7, 0.4], 0.4, BLUE),
                   Sphere([0.6, -0.8, 0.5], 0.5, YELLOW),
                   Sphere([-1.2, 0.2, 0.5], 0.5, MAGENTA),
                   Sphere([-1.7, -0.5, 0.3], 0.3, GREEN),
                   Sphere([-2.0, 1.31, 1.3], 1.3, RED)]

        planes = [Plane([5, 0, 0], [0, 0, 1], GREY)]

        return Scene(lights, spheres, planes)
