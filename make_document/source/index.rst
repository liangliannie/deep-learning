.. badblock documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:42:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deep learning tutorial by building U-Net in Pytorch 
===================================================================================================

Deep learning has been fast developed in recent years, which triggers lots of researches in medical image analysis. In this tutorial, I will show you how to improve the quality of image(or say image denoising/impainting) by using one of the many networks, UNet.


Reasons for this tutorial 
----------------------------------
I started my internship in Siemens, Knoxville at the begining of 2019, where I first touched and learned deep learning. 
Great thanks to super nice manager(BILL) since I learn a lot from him!

During my internship, I try to use deep learning(UNet, FrameletNet, GAN) to do image impainting or denoising, while I think this is so great a chance for me to learn about deep learning and explore different kinds of networks, and this makes me feel excited almost every day so I want to share this with you. 

In fact, my coworkers are also very interested in building or learning neural networks while their think that deep learning is very hard stops them. And thus, I also mean to write this tutorial for people who are in the field of medical imaging who might want to build their own deep learning networks but they do not know how/where to start. 

To be honest, in either deep learning or machine learning, the optimal choices for the most suitable parameters/models are always hard to make and it is also consuming most of the time, but initiating and building deep learning network and making it work is far more simple than you thought!


Last but not least, I am still a beginner and learner in deep learning/matchine learning, and please point/teach me out if I will be wrong in the following content.

Pipeline for this tutorial 
----------------------------------
Here, let's begin with the following content to open your new world to the deep learning(AI)!

   First, we will install all the necessary packages/softwares for the deep learning(it will be great if you have a GPU), and in this tutorial we will utilize Pytorch.

   Next, we will talk about how to precesss your data (so important), such as shaping and normalization! And I will always try to add a method named augumator to further enlarge our datasets in case that the training data is very small. 

   Third, we will go over neural network(U-Net), which will let you learn the core of deep learning. In this part, I will briefly introduce the idea and the structure of U-Net. 

   Finally, we will train our U-Net with our training datasets and use visualization tool to view our results.

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
