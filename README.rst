TPU installation notes
----------------------

Use Python 3.6.

Assuming Ubuntu 18.04, install packages::

   sudo apt install libopenblas-base libomp5

Install wheels for torch xla 0.5:

- gs://tpu-pytorch/wheels/torch-0.5-cp36-cp36m-linux_x86_64.whl
- gs://tpu-pytorch/wheels/torch_xla-0.5-cp36-cp36m-linux_x86_64.whl
- gs://tpu-pytorch/wheels/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

Create TPU node using pytorch0.5 version (double-check zone and size),
use e.g. 172.16.0.0 as IP mask.

Export env variable (check TPU address if different form 172.16.0.2)::

    export XRT_TPU_CONFIG="tpu_worker;0;172.16.0.2:8470"

