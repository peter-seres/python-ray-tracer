# ray_tracing
Ray Tracing with python using Nvidia GPU (work in progress)


I have been implementing a ray tracing project in python where my aim is 
to achieve a >30 fps performance for complex scenes. The rendering is 
done via a CUDA kernel defined using Numba CUDA.

**Dependencies:** 
* python 3.7
* numpy
* PIL 
* numba
* compatible cuda toolkit installation with an Nvidia GPU

**Current Usage:** Run main.py which will produce an output.png image and will 
print the render time. Define the scene (spheres and lights) using a few 
dictionaries in generate_scene() within main.py
