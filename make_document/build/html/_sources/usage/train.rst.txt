.. badblock documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:42:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Train your network
====================================

In this part, we will focus on how to train our UNet. We will first cover several common loss functions and then go with how to train. Finally, we also give a method to further improve the training speed.

Loss functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The loss function is the function to map an event or values of one or more to prepresent the cost associated with the event. And if we view the deep learning as an optimization problem, our goal will be to minimize the loss function. 

To be simple, the loss function is a method to evaluate how well our neural network is and we use backward proprogation to change the weights in the network to further improve our neural network or say to minimize our cost/loss.

Here, I will enumerate several common used loss functions based on Pytorch.

L1Loss
"""""""""""""""""""

L1Loss creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target y.

.. code-block :: Python

   torch.nn.L1Loss()

.. math::

   l(x,y) = L = \{l_1, ..., l_N\}^T, l_n = |x_n-y_n|

where N is the batch size.

MSELoss
"""""""""""""""""""

MSELoss creates a criterion that measures the mean squared error (squared L2 norm) between each element in the input x and target y.

.. code-block :: Python

   torch.nn.MSELoss()

.. math::

   l(x,y) = L = \{l_1, ..., l_N\}^T, l_n = (x_n-y_n)^2

MSSIM
"""""""""""""""""""

MSSIM is the structural similarity(SSIM) index, which predicts the perceived quality of digital television and cinematic pictures, as well as other kinds of digital images and videos. To be simple, SSIM is used for measuring the similarity between two images, while it ranges in [0,1]. SSIM is designed to improve the traditional methods, while it works well while being combined with traditional methods, such as L1, L2, in deep learning.



.. code-block :: Python

   MSSSIM(window_size=9, size_average=True)


Combinations
"""""""""""""""""""

Try a combination of different loss functions.

.. code-block :: Python

   Loss = L1Loss + MSSSIM
   Loss = L1Loss + MSELoss + MSSSIM


.. note::
   
   Here, only three loss functions are listed, and there are still a lot available there online. Google it! You can even define you own loss function based on your task. Network is flexible, and so with the loss functions. DIY!


Train, Train, Train
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We start with the optimizer and then the structure of training.

Optimizer
"""""""""""""""""""

The optimizer is the package implementing optimization algorithms. Most commonly used methods are already supported in Pytorch, and we can easily construct an optimizer object in pytorch by using one line of code. Two examples of optimizers are listed below.

.. code-block:: python

	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	optimizer = optim.Adam([var1, var2], lr=0.0001)

where the parameters of the optimizer can be set, such as the learning rate, weight decay and so on when initiating the optimizer.

From my view, the optimizer takes job of the backward propagation and optimizes the neural network based on certain optimization algorithm. The details of pytorch optim can be found `<https://pytorch.org/docs/stable/optim.html>`_ . To be simple, the optimizer change the weights in neural network to make it the optimal weights for mapping the input to target.

Backward Propagation
"""""""""""""""""""""""""

Training has been made really simple in Pytorch::

	for epoch in range(2):  # loop over the dataset multiple times

	    running_loss = 0.0
	    for i, data in enumerate(trainloader, 0):
		# get the inputs
		badsino, goodsino = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = Unet(badsino)
		loss = criterion(outputs, goodsino)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
		    print('[%d, %5d] loss: %.3f' %
		          (epoch + 1, i + 1, running_loss / 2000))
		    running_loss = 0.0

	print('Finished Training')


Hot Start?(Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Try to restart your optimizer::

            if epoch % next_reset == 0:
                print("Resetting Optimizer\n")
                optimizer = torch.optim.Adam(network.network.parameters(), lr=self.opts.initial_lr, betas=(0.5, 0.999))
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.opts.lr_decay)
                network.set_optimizer(optimizer)

                # Set the next reset
                next_reset += self.opts.warm_reset_length + warm_reset_increment
                warm_reset_increment += self.opts.warm_reset_increment


