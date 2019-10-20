TPU Pipeline for Kaggle RSNA Intracranial Hemorrhage Detection
==============================================================

See https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/

This is a very simple pipeline, my main goal was to check if
Pytorch TPU support is good enough for classification.

**WARNING:** this is not a complete pipeline yet, only training
"works" (loss is going down, nothing checked yet).

I recommend checking
https://github.com/pytorch/xla and reading through
https://github.com/pytorch/xla/blob/master/API_GUIDE.md
if you want to try using TPU yourself.

Performance
-----------

Tested on:

- 2080ti with AMD Ryzen 7 3700X 8-Core Processor, in float32
- TPUv2 with n1-standard-16 VM

Input resolution is 448x448, optimizer is SGD.
``amp 01`` below stands for 01 level in apex amp (automatic mixed precision).

======  =========  ==========  ==========  =======  ========
Device  Precision  Network     Batch size  s/batch  images/s
======  =========  ==========  ==========  =======  ========
TPUv2   float32    resnet50    16x8        0.289    443
2080ti  float32    resnet50    16          0.226    71
TPUv2   float32    resnet50    24x8        0.392    490
2080ti  float32    resnet50    24          0.349    69
2080ti  amp O1     resnet50    24          0.228    105
2080ti  amp O1     resnet50    32          0.285    112
TPUv2   bfloat16   resnet50    24x8        0.342    561
2080ti  amp O1     resnet101   24          0.356    67
TPUv2   float32    resnet101   24x8        0.514    236
TPUv2   bfloat16   resnet101   24x8        0.514    374
======  =========  ==========  ==========  =======  ========

TODO:

- max batch sizes
- heavier models
- TPUv3

Note: resnet50 is definitely either I/O or CPU bound or both.

Overall impressions
-------------------

So far working with TPU looks very similar to working with a multi-GPU with
distributed data parallel - it needs about the same amount of modifications,
maybe even smaller, at least when all ops are supported and shapes are static,
like it is for a simple classifications task.
It also needs an efficient data pipeline and
a powerful machine to feed all 8 TPU core,
similar to what you'd need for an 8-GPU machine.
So far I didn't see any strange hangs or stability issues.

In terms of the API, what I really like about pytorch TPU support:

- you can use any pytorch models as long as all ops are supported, you don't
  need to convert any weights to/from TPU
- TPU-specicif API is quite small and clear:
  https://github.com/pytorch/xla/blob/master/API_GUIDE.md
- you can use the same data pipeline as for a regular GPU

With TF, recommended approach is to prepare and feed TFRecords which are read
by the TPU from Google Cloud Storage. On one hand, it looks much less convenient
for quick development and prototyping (e.g. probably you won't be able to use
you favorite augmentations library). On the other hand, you don't need
a powerful machine to feed the data.

In terms of price effectiveness, it's a hard call and depends very much on
how much does TPU cost you and where can you rent GPUs. I also didn't check
preemptible TPUs yet.

TPU installation notes
----------------------

Make sure to use a VM which is large enough, at least
``n1-standard-16`` to not be VM compute/network bound
(also you can run out of memory on a smaller VM).

Check installation options at https://github.com/pytorch/xla

Alternatively, you can start a VM with Ubuntu 18.04 and install
pre-built wheels - this was the most comfortable approach for me.

Assuming Ubuntu 18.04, install packages::

   sudo apt install libopenblas-base libomp5

Install wheels for torch xla 0.5, you can copy them with ``gsutil cp``:

- ``gs://tpu-pytorch/wheels/torch-0.5-cp36-cp36m-linux_x86_64.whl``
- ``gs://tpu-pytorch/wheels/torch_xla-0.5-cp36-cp36m-linux_x86_64.whl``
- ``gs://tpu-pytorch/wheels/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl``

Create TPU node using pytorch-0.5 version (double-check zone and size,
also VM should be in the same zone), use e.g. 172.16.0.0 as IP mask.
Code was tested with TPU-v2 so far, but should work with TPU-v3 as well.

Export env variable (check TPU address if different form 172.16.0.2)::

    export XRT_TPU_CONFIG="tpu_worker;0;172.16.0.2:8470"

And also enable bfloat16::

    export XLA_USE_BF16=1

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

License
-------

License is MIT.
