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


@dataclass
class Triangle:
    V0: Vector3D
    V1: Vector3D
    V2: Vector3D
    color: Color

    data_length: int = 12

    def to_array(self) -> np.ndarray:
        data = np.zeros(self.data_length, dtype=np.float32)
        data[0:3] = np.array(self.V0)
        data[3:6] = np.array(self.V1)
        data[6:9] = np.array(self.V2)
        data[9:12] = self.color

        return data


class Scene:
    def __init__(self, lights: List[Light], spheres: List[Sphere], planes: List[Plane], triangles: List[Triangle]):
        self.lights = lights
        self.spheres = spheres
        self.planes = planes
        self.triangles = triangles

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

    def get_triangles(self) -> np.ndarray:
        """ Generate data array containing sphere data: """
        data = np.zeros((Triangle.data_length, len(self.triangles)), dtype=np.float32)

        for i, l in enumerate(self.triangles):
            data[:, i] = l.to_array()

        return data

    def generate_scene(self) -> (np.ndarray, np.ndarray, np.ndarray):
        return self.get_spheres(), self.get_lights(), self.get_planes(), self.get_triangles()
