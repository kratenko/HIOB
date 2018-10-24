HIOB
====
A modular hierarchical object tracking framework

# Installation

#### get HIOB
clone the repositiory

    $ git clone https://github.com/kratenko/HIOB.git

or

    $ git clone git@github.com:kratenko/HIOB.git

#### virtual environment
HIOB needs python3 and tensorflow. We recommend building a virtual environment for HIOB.
Build a virtual environment somewhere outside of the HIOB directory and activate it:

    $ virtualenv -ppython3 hiob_env
    $ source hiob_env/bin/activate
    
#### dependencies
Install required packages:

    # for using your GPU and CUDA
    (hiob_env) $ cd HIOB
    (hiob_env) $ pip install -r requirements.txt

This installs a tensorflow build that requires a NVIDIA GPU and the CUDA machine learning library. You can alternatively use a tensorflow build that only uses the CPU. It should work, but it will not be fast. We supply a diffenrent requirements.txt for that:

    # alternatively for using your CPU only:
    (hiob_env) $ cd HIOB
    (hiob_env) $ pip install -r requirements_cpu.txt

# Run the demo
HIOB comes with a simple demo script, that downloads a tracking sequence (~4.3MB) and starts the tracker on it. Inside your virtual environment and inside the HIOB directory, just run:

    (hiob_env) $ ./run_demo.sh
    
If all goes well, the sample will be downloaded to `HIOB/demo/data/tb100/Deer.zip` and a window will open that shows the tracking process. A log of the tracking process will be created inside `HIOB/demo/hiob_logs` containing log output and an analysis of the process.
