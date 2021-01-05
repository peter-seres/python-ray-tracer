The `pycuda` package has to be compiled from source with OpenGl enabled. 

#### PyCuda repo:

https://github.com/inducer/pycuda

git@github.com:inducer/pycuda.git

#### Enable OpenGl interop

`./configure.py --cuda-enable-gl`

`python setup.py install`


#### Make sure the following exe can be found:

Add to PATH:

`C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.27.29110\bin\Hostx64\x64`

or something similar.

#### Test

```
import pycuda.driver as drv
import pycuda.gl as pycudagl
```

### OPENGL

pip install pyopengl doesn't work because it doesn't install GLUT properly

fetch the wheel files from:

https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl

choose the appropriate file and install using:

`pip install <filename>.whl`

