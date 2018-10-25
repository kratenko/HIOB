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
    
If all goes well, the sample will be downloaded to `HIOB/demo/data/tb100/Deer.zip` and a window will open that shows the tracking process. A yellow rectangle will show the position predicted by HIOB and a dark green frame will show the ground truth included in the sample sequence. A log of the tracking process will be created inside `HIOB/demo/hiob_logs` containing log output and an analysis of the process.


# Getting more test samples
## The tb100 online tracking benchmark
The deer example used in the demo is taken from the tb100 online benchmark by *Yi Wu* and *Jongwoo Lim* and *Ming-Hsuan Yang*. The benchmark consists of 98 picture sequences with a total of 100 tracking sequences. It can be found under http://visual-tracking.net/ HIOB can read work directly on the zip files provided there. The benchmark has been released in a paper:  http://faculty.ucmerced.edu/mhyang/papers/cvpr13_benchmark.pdf

Since the 98 sequences must be downloaded individually from a very slow server, the process is quite time consuming. HIOB comes with a script that can handle the download for you, it is located at `bin/hiob_downloader` within this repository. If you call it with argument `tb100` it will download the whole dataset from the server. This will most likely take several hours.

## The Princeton RGBD tracking benchmark
HIOB also works with the [Princeton Tracking Benchmark](http://tracking.cs.princeton.edu) and is able to read the files provided there. That benchmark provides depth information along with the RGB information, but the depth is not used by HIOB. Be advised that of the 100 sequences provided only 95 contain a ground truth. The original implementation of HIOB has been evaluated by the benchmark on April 2017, the results can be seen on the [evaluation page](http://tracking.cs.princeton.edu/eval.php) named `hiob_lc2`.
