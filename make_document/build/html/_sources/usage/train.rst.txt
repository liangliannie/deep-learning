.. badblock documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:42:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Train your network
====================================

In this part, I will try to elaborate what is UNet and how it works.

Loss functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Try several different loss functions to obtain the best results, here we have used::

   torch.nn.L1Loss()
   torch.nn.MSELoss()
   MSSSIM(window_size=9, size_average=True)

Or try a combination of them.


Train?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Hot Start?
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

.. toctree::
   :maxdepth: 2
      
   /usage/train_file.rst
