.. badblock documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:42:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Adjust your parameters!
====================================

In this part, I will try to explain about the problems I have confronted with and the parameters I have tuned when I try to train the neural network and also the methods I have found to view or fixed the problems. There must be a lot of other problems I can not cover here, and in this case google it first and see if other people have solved it.


About memory and time consumption
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Memory 
""""""""""""""""""""""
Let's begin to view the usage[memory consumption] of GPU first, which can be easily done by calling the following code in terminal.

.. code-block:: console

   nvidia-smi
   watch -n 1 nvidia-smi 

Where nvidia-smi will show the results one time, by using `watch -n 1 nvidia-smi` we can let the terminal update the results every 1 second. One sample of the results are shown below.

.. image:: memory.png
    :width: 500px
    :align: center
    :alt: Unet

.. note:: 
   
   Keeping an eye on the memory consumption can help us do the memory management during the traing. For example, if our peak usage of memory is too high, we better reduce some parameters in training. e.g. the size of data input, the batchsize of data input, even the size of network[the width or depth of the neural network].

Time and memory 
""""""""""""""""""""""

We can also check the memory consumption by using python wrapper. In the following, I have listed one example based on the `profile` function from memory profiler package, where both the time and memory consumption of program can be monitored by adding a wrapper in front of our target function.


.. code-block:: python

   from memory_profiler import profile

   @profile()
   def basic_mean(N=5):
      nbrs = list(range(0, 10 ** N))
      total = sum(nbrs)
      mean = sum(nbrs) / len(nbrs)
      return mean

   if __name__ == '__main__':
       basic_mean()

After running 

.. code-block:: console

   python -m memory_profiler profile.py

the log will be generated and shown in the terminal.

.. code-block:: console

   Line #    Mem usage    Increment   Line Contents
   =================================================
     2     36.7 MiB     36.7 MiB   @profile()
     3                             def basic_mean(N=5):
     4     40.5 MiB      3.7 MiB       nbrs = list(range(0, 10 ** N))
     5     40.5 MiB      0.0 MiB       total = sum(nbrs)
     6     40.5 MiB      0.0 MiB       mean = sum(nbrs) / len(nbrs)
     7     40.5 MiB      0.0 MiB       return mean


If we want to have the time-based memory usage, such as a report, we can run 

.. code-block:: console

   mprof run <executable>
   mprof plot

In this way, a recorded file with the time-based memory usage will be generated as following,

.. image:: MemoryManagement.png
    :width: 700px
    :align: center
    :alt: Unet

This is super cool, right?! But for real time checking, `watch -n 1 nvidia-smi` works better.

.. note::

   There must be other methods, which I can not enumerate, please keep an eye on them too.


Next, let's go into more parameters dealing with not only the memory and time consumption but also the learning accuracy.


Parameters in data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If we are working on image denoising, the input are images with noise.
In this case, the parameters we can change include but not limited to [the shape], [the distribution], [range/maximum/minimum/mean/median/deviation] of data. In this part, I would mainly focus on two of them, the shape and the distribution respectively.


Shape of data
"""""""""""""""""""""""""
When I am talking the shape of data, I mean the size of image. For example, each input of the neural network might be 3*256*256, which is a three-channel image with both width and height as 256. A bigger size of image will surely cost more operations and further affect the learning speed. Hence, choosing the right size or deciding the right resolution will be the first thing to track at the beginning or during the training.

E.g., we can scale the origin 3*256*256 to grayscale as 1*256*256, and we can further reduce the height and width by sampling origin image by 4:1 and obtain images with the size of 1*64*64.

.. note:: 
   
   The options are flexible based on different targets. 

Distribution of data
"""""""""""""""""""""""""

I have tried several most and found that mapping the range of origin images to [0,1] can achive my best performance. But the case might be different based on different tasks.


Normalization, as another method, is a technique often applied. The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values. Normalization is required only when features have different ranges. 

.. note::

   For more details, please refer to `<https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029>`_, which is a good article about normalization.


Parameters in optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Deep learning neural networks are trained using the stochastic gradient descent, which is an optimization algorithm that estimates the error gradient for the current state of the model using examples from the training dataset, then updates the weights of the model using the back-propagation of errors algorithm, referred to as simply backpropagation. Regarding to the otpimization, we will include two parameters which I have change frequently, inital learning rate and learning decay rate.




Inital Learning Rate 
"""""""""""""""""""""""""

The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated. The learning rate may be the most important hyperparameter when configuring your neural network. Therefore it is vital to know how to investigate the effects of the learning rate on model performance and to build an intuition about the dynamics of the learning rate on model behavior. For more details, please refer to `<https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/>`_.

.. note::
   Overall, a learning rate that is too large can cause the model to converge too quickly to a suboptimal solution, whereas a learning rate that is too small can cause the process to get stuck.





Learning Decay Rate 
"""""""""""""""""""""""""

Learning Decay Rate is the amount that the learning rate are updated during training is referred to as the step size or the “learning rate.”  The learning decay rate is usually initated with a big value and then decay to a small value based on different algorithms. The reason for this is to make the weights of the network converge quickly to the solution at the begining and then converge more slow to reach the optimal solution when it is close to the solution. 

Parameters in neural network
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

last but not the last, we can also tune parameters in neural network, UNet. Since most layers in UNet are equipped as the convolutional layer, then we can change the kernel size which can be managed to reduce the weight Convnets while making them deeper.



Kernel Size
"""""""""""""""""""""""""

The number of weights is dependent on the kernel size instead of the input size which is really important for images. Convolutional layers reduce memory usage and compute faster. For more, this article is a good reference -> `<https://blog.sicara.com/about-convolutional-layer-convolution-kernel-9a7325d34f7d>`_


Other Parameters
"""""""""""""""""""""""""
There are also other things we could try, such as adding bias to fully connect layer, changing the number of upsampling and downsampling, trying different activation functions, revising the layers in up/down blocks and so on. 


Try other neural networks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

UNet is one of many neural networks for computer vision, and there are a lot of other networks online with open source models available in github. For example, Facebook research has post several modern models in `<https://github.com/pytorch/vision/tree/master/torchvision/models>`_ which include alexnet, densenet, googlenet, mobilenet, resnet, vgg. What we need to do is to change/revise certain parts of the networks and make it meet our requirements, and then the parameters tuning would be similar to what we have discussed above.

.. note::

   Never stop in learning, because this area has been developed so fast. 

Good Luck!


Liang, May 30, 2019



  


