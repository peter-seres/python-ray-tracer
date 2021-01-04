# ray_tracing
This repository demonstrates ray tracing implemented in python using CUDA and the JIT compiler of numba.
The following scene takes less than 1 ms to render after the code is compiled (using an Nvidia Quadro P1000)

![Render example](https://user-images.githubusercontent.com/27952562/62824447-2d416c80-bb9e-11e9-8a35-cd432c76c976.png)

### CUDA Toolkit installation
A compatible installation of the CUDA toolkit is required.

Download from:
[https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)


### Usage

- `main.py` will produce a PNG image in the `/output/` folder.
- (...) defining a scene (at the moment can be done using the Scene class. Check out `scene.py`)

### Feature to-do list
- more robust way of defining materials and shader settings
- refractions
- scatter
- anti aliasing
- cuda - open gl interop real-time display
- gui for scene management
