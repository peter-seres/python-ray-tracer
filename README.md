# ray_tracing
Ray Tracing with python using GPU (work in progress)


I have been implementing a ray tracing project in python where my aim is 
to achieve a >30 fps performance for complex scenes. The rendering is 
done via a CUDA kernel defined using Numba CUDA.

The following scene takes less than 1 ms to render after the code is compiled (laptop with an Nvidia Quadro P1000)

![Render example](https://user-images.githubusercontent.com/27952562/62824447-2d416c80-bb9e-11e9-8a35-cd432c76c976.png)

**Dependencies:** 
* python 3.7
* numpy
* PIL 
* numba
* compatible cuda toolkit installation with an Nvidia GPU

**Current Usage:** Run main.py which will produce an output.png image and will 
print the render time. Define the scene (spheres and lights) using a few 
dictionaries in generate_scene() within main.py


**To do list**
1) add tracing with multiple light sources
2) refraction
3) scatter
4) moving and rotating the camera
5) window view and real-time renderer
6) gui to place elements into the scene and change material types
