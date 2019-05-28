.. badblock documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:42:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Install necessary packages
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Python is a programming language that lets you work quickly and integrate systems more effectively. From my point, python is simple to use and learn. The reason why we use python is that currently most deep learning frameworks have already been implemented based on python and plenties of open source packages that are available in the field of data science can be utilitized for our data analysis, such as scipy, scikit-learn, pandas. Pythonic makes life easier.

Install Python
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Direct install Python from *Python* 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Python2 will not be supported any more by the community, and hence let's work on Python3 and install the latest python. The latest Python can be downloaded and installed from `<https://www.python.org/downloads/>`_ . I have installed 3.6, please try to install a version above 3.6. 

Indirect install Python from *Conda* 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Conda is an open source package management system and environment management system that runs on Windows, macOS and Linux. Conda quickly installs, runs and updates packages and their dependencies. 

Conda can be found via `<https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.

Conda is recommended when multiple versions of python will be installed in the same system, and the version of python can be changed easily by using:

.. code-block:: console

   source activate myenv


For example, 

.. code-block:: console

   source activate py36

with the whole list of pythons with different versions can be shown as:

.. code-block:: console
   
   conda env list

other useful methods for install certain packages including

.. code-block:: console
   
   conda search scipy
   conda install --name myenv scipy
   conda install scipy=0.15.0
   source deactivate
   conda info --envs
   conda list -n myenv scipy

Install Pytorch with CUDA
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before we start with pytorch, please make sure CUDA has been installed, where
CUDA is a parallel computing platform and application programming interface (API) model created by Nvidia.

Install CUDA Driver
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Download and install your cuda driver from `<https://developer.nvidia.com/cuda-downloads?target_os=Linux>`_.

.. note:: 
   Please check the same link for updating with the latest driver. A good match of driver with the GPU will largerly increase the speed, so always make sure you have the latest driver with your GPU.


Install pytorch 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pytorch is an open source deep learning platform that provides a seamless path from research prototyping to production deployment.

pytorch can be found via `<https://pytorch.org/get-started/locally/>`_.


Other packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a list of other packages that are optional to install, while most could be install by using **pip**. [if you are under conda, make sure you are using the correct **pip** under the correct version of python]

* $ pip install visdom
* $ pip install numpy
* $ pip install matplotlib
* $ pip install Pillow
* $ pip install scipy
* $ pip install Augmentor


Plus, if you would consider mssim loss too, please include 'pytorch_msssim' folder{we will talk about this later} and if there are other package needed, try **pip**

.. code-block:: console
   
   pip install packagename


NOW, you are all set with all the packages needed for the deep learning Unet, and in the next step we will forward to prepare our data.

Build in Docker (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If GPU or the practical hardware is unavailable, we can also learn the deep learning by utilizing all the cloud services availble, such as AWS. Here, a simple system Docker is covered for the cases where we need to run deep learning in a server. 

Docker is a computer program that performs operating-system-level virtualization. The advantage of Docker is that we can only install the packages we need and then the extra cost of unnecessary components in the operating system can be reduced.

.. seealso::
   
   If a python environment with pytorch is hard to obtain locally, Docker is always a good choice to make your network run in cloud. Note: a most recent pytorch with NVIDIA can be pulled from `<https://docs.nvidia.com/deeplearning/dgx/pytorch-release-notes/running.html>`_. 

Docker is simple to use too! The followings are a summary of Docker codes for reference.

.. code-block:: console

   # List Docker images
        docker image ls

   # List Docker containers (running, all, all in quiet mode)
	docker container ls
	docker container ls --all
	docker container ls -aq
	docker container stop xxxxxxx
   # List Docker containers (running, all, all in quiet mode)
	docker build -t friendlyhello .  # Create image using this directory's Dockerfile
	docker run -p 4000:80 friendlyhello  # Run "friendlyname" mapping port 4000 to 80
	docker run -d -p 4000:80 friendlyhello         # Same thing, but in detached mode
	docker container ls                                # List all running containers
	docker container ls -a             # List all containers, even those not running
	docker container stop <hash>           # Gracefully stop the specified container
	docker container kill <hash>         # Force shutdown of the specified container
	docker container rm <hash>        # Remove specified container from this machine
	docker container rm $(docker container ls -a -q)         # Remove all containers
	docker image ls -a                             # List all images on this machine
	docker image rm <image id>            # Remove specified image from this machine
	docker image rm $(docker image ls -a -q)   # Remove all images from this machine
	docker login             # Log in this CLI session using your Docker credentials
	docker tag <image> username/repository:tag  # Tag <image> for upload to registry
	docker push username/repository:tag            # Upload tagged image to registry
	docker run username/repository:tag                   # Run image from a registry


  
Using Docker within DGX
""""""""""""""""""""""""""""""""""""""

**Special Requirement for DGX user:** In order to connect to his Docker daemon a user has to commit the parameter "-H unix:///mnt/docker_socks/<user_name>/docker.sock" with every Docker command.

* e.g. "docker -H unix:///mnt/docker_socks/<user_name>/docker.sock run --rm -ti <image_name> [optional_command]"
* e.g. "docker -H unix:///mnt/docker_socks/<user_name>/docker.sock image ls" alternatively use the script "run-docker.sh" in /usr/local/bin:
* e.g. "run-docker.sh --rm -ti [further_options] <image_name> [optional_command]"

I was using a line like below in a bash file to let the docker run within DGX by using slum.

.. code-block:: console

   sbatch --gres=gpu:1 --mail-user=liang.li.uestc@gmail.com --mail-type=ALL --output=/badblock/li_dataset/log.txt --job-name=trainbd --error=/badblock/li_dataset/error.txt run-nvidia-docker.sh --rm --name train_unet -v /badblock/li_dataset/:/badblock/ -w /badblock/ nvcr.io/nvidia/pytorch:19.01-py3 python /badblock/code/badblock-fillgaps/train_unet.py --training-file="/badblock/data/DataTOF_Train/" --test-file="/badblock/data/DataTOF_test/" --mask-file="/badblock/data/mask.pkl" --output-path="/badblock/output/"



