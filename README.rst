TPU Pipeline for Kaggle RSNA Intracranial Hemorrhage Detection
==============================================================

See https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/

This is a very simple pipeline, my main goal was to check if
Pytorch TPU support is good enough for classification.

This is no substitute for reading the docs, I recommend checking
https://github.com/pytorch/xla and reading through
https://github.com/pytorch/xla/blob/master/API_GUIDE.md
if you want to try using TPU yourself.

TPU installation notes
----------------------

Check installation options at https://github.com/pytorch/xla

Alternatively, you can start a VM with Ubuntu 18.04 and install
pre-built wheels - this was the most comfortable approach for me.

Assuming Ubuntu 18.04, install packages::

   sudo apt install libopenblas-base libomp5

Install wheels for torch xla 0.5, you can copy them with ``gsutil cp``:

- gs://tpu-pytorch/wheels/torch-0.5-cp36-cp36m-linux_x86_64.whl
- gs://tpu-pytorch/wheels/torch_xla-0.5-cp36-cp36m-linux_x86_64.whl
- gs://tpu-pytorch/wheels/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl

Create TPU node using pytorch-0.5 version (double-check zone and size),
use e.g. 172.16.0.0 as IP mask.

Export env variable (check TPU address if different form 172.16.0.2)::

    export XRT_TPU_CONFIG="tpu_worker;0;172.16.0.2:8470"

General installation
--------------------

Use Python 3.6.

Install requirements and the package::

    pip install -r requirements.txt
    python setup.py develop

Running
-------

Place competitiono data into ``./data`` folder::

    $ ls data
    stage_1_sample_submission.csv  stage_1_test_images  stage_1_train.csv  stage_1_train_images

Run training (TPU is used by default, single-GPU is also supported via ``--device=cuda``)::

    python -m rsna.main
