import numpy as np

ray_origin = np.array([0, 0, 0])
ray_direction = np.array([1, 0, 0])

sphere_origin = np.array([100, 0, 0])
sphere_radius = 10.0

# Ray intersect sphere:

p_0 = np.array([0, 0, 0])                   # ray origin
d = np.array([1, 10, 0])                     # ray direction

c = np.array([100, 0, 0])                   # sphere origin
r = 10.0                                    # sphere radius

A = np.dot(d, d)

B = 2 * np.dot(d, p_0-c)

C = np.dot(p_0-c, p_0-c) - r*r

discriminant = B**2 - 4 * A * C

print(f"A: {A}, B: {B}, C: {C}")
print(discriminant)
