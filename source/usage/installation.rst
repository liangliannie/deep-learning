.. badblock documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:42:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Installation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Install Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install Python from *Python* 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The latest Python can be downloaded and installed from `<https://www.python.org/downloads/>`_ , a version above 3.6 is recommended. 

Install Python from *Conda* 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Conda can be found via `<https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

Conda is also recommended since many different versions of python could be changed based on:

.. code-block:: console

   source activate myenv

with the whole list of pythons with different versions can be shown as:

.. code-block:: console
   
   conda env list



Install Pytorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before start with pytorch, make sure CUDA has been installed.
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Install your cuda from `<https://developer.nvidia.com/cuda-downloads?target_os=Linux>`_.

.. note:: 
   Check here to update with the latest driver. A good match of driver with the GPU will largerly increase the speed, thus please make sure you have the latest driver with your GPU.


Install pytorch 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

pytorch can be found via `<https://pytorch.org/get-started/locally/>`_.


Other packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a list of other packages that are optional to install, while most could be install by using **pip**. [if you are under conda, make sure you are using the correct **pip** under the correct version of python]

* $ pip install visdom
* $ pip install numpy
* $ pip install matplotlib
* $ pip install Pillow

Plus, please include 'pytorch_msssim' folder if you want to use msssim as a loss, and if there are other package needed, try **pip**

.. code-block:: console
   
   pip install packagename


NOW, you are all set with all the packages need for the deep learning Unet, and next we will forward to prepare our data.


.. seealso::
   
   If a python environment with pytorch is hard to obtain locally, Docker is always a good choice to make your network run in cloud. Note: a most recent pytorch with NVIDIA can be pulled from `<https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html>`_. 




  



