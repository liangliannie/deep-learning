.. badblock documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:42:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deep learning tutorial by building U-Net in Pytorch
===================================================================================================

Deep learning has been fast developed in recent years, which triggers lots of researches in medical image analysis. In this tutorial, I will show you 
how to improve the quality of image(or say image denoising) by using one of the many networks, UNet.


Reasons for this tutorial 
----------------------------------
I started my internship in Siemens, Knoxville at the begining of 2019, where I first touched and learned deep learning. 
Great thanks to super nice manager(BILL) since I do learn a lot from him!! Please note that I am also a beginner in deep learning, and thus point me out if I will be wrong in the following content.

During my internship, I try to use deep learning(UNet, FrameletNet, GAN) to do image impainting and denoising, while I think this is so great a chance for me to learn about deep learning and explore different kinds of networks, and this makes me feel excited almost every day so I want to share this with you. 

In fact, a lot of my coworkers are also very interested in building or learning about deep learning while they think
deep learning is very hard to learn. Hence, I also write this tutorial for people who are in the field of medical imaging
or other who might want to build their own deep learning networks but they do not know how to start. To be honest, to select the most suitable parameters/models
is the most time consuming part of deep learning, but to start deep learning is far more simple than you thought!

Goals for this tutorial 
----------------------------------
Hence, lets begin with this to open your new world to the deep learning(AI)!

   First, we will try to install all the necessary packages for the deep learning, and it will be great if you have a GPU.

   And then, we will talk about how to precesss your data (so important)! 

   Third, a quick start with U-Net will lead you to the heart of deep learning.

   Finally, we will train our U-Net and use several visualizations to view our results.

.. toctree::
   :maxdepth: 2
   :caption: Here all the contents go:

   usage/installation.rst
   usage/data.rst
   usage/quickstart.rst
   usage/train.rst
   usage/view.rst
   usage/adjust.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
