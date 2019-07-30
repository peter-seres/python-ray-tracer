from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from camera import Camera


fig = plt.figure()
ax = Axes3D(fig)

camera = Camera(5, 5, field_of_view=60)

ax.scatter(camera.origin[0], camera.origin[1], camera.origin[2])

for j, k, ray, pix_loc_global in camera.generate_rays():
    ax.scatter(pix_loc_global[0], pix_loc_global[1], pix_loc_global[2], marker='^')
    ax.quiver(pix_loc_global[0], pix_loc_global[1], pix_loc_global[2], ray.dir[0], ray.dir[1], ray.dir[2],
              length=0.5, arrow_length_ratio=0.2)

c1, c2, c3 = camera.get_local_coord_system()

ax.quiver(camera.origin[0], camera.origin[1], camera.origin[2], c1[0], c1[1], c1[2], length=0.5, normalize=True)
ax.quiver(camera.origin[0], camera.origin[1], camera.origin[2], c2[0], c2[1], c2[2], length=0.5, normalize=True)
ax.quiver(camera.origin[0], camera.origin[1], camera.origin[2], c3[0], c3[1], c3[2], length=0.5, normalize=True)


ax.set_aspect(aspect='auto')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
