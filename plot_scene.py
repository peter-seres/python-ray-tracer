def plot(scene):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = Axes3D(fig)

    # Plot camera coordinate system:
    for x, y, z, i, j, k in scene.camera.enum_coordinate_system():
        ax.quiver(x, y, z, i, j, k, length=0.2)

    # Plot spheres in the scene:
    for obj in scene.objects:
        ax.scatter(obj.origin[0], obj.origin[1], obj.origin[2], s=5)

    # # Plot pixel locations:
    for corner in scene.camera.generate_corners():
        ax.scatter(corner[0], corner[1], corner[2], s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
