.. badblock documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:42:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Visualize your results!
====================================


Install Visdom and Open server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Visdom is a flexible tool for creating, organizing, and sharing visualizations of live, rich data. Supports Torch and Numpy, which can be found `<https://github.com/facebookresearch/visdom/>`_.

From my view, visdom is similar to the TensorBoard from Tensorflow while it is still under developing I hope it will be much more strong in the future. 

Install 
"""""""""""""""""""""""""
Visdom can be easily installed by using pip.

.. code-block:: python

   	pip install visdom

There are also other methods to install visdom which I am not familiar!

.. code-block:: python

   # Install Torch client
   # (STABLE VERSION, NOT ALL CURRENT FEATURES ARE SUPPORTED)
   luarocks install visdom
   # Install visdom from source
   pip install -e .
   # If the above runs into issues, you can try the below
   easy_install .

   # Install Torch client from source (from th directory)
   luarocks make

Open server 
"""""""""""""""""""""""""
After install visdom, you can start server from command line by running

.. code-block:: python

   python -m visdom.server

Then, the visdom can be accessed by going to `http://localhost:8097 <http://localhost:8097>`_ in your browser, or your own host address if specified.


Project your output to Visdom
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now you have installed visdom, and next we will work on project our output to the server and then we can view them.


Step by step 
"""""""""""""""""""""""""

First, we need to initate our visdom object by::

   vis = visdom.Visdom()

Second, we need to open one window project our output::
    
   window = None

Last, we need update our window with the output::

   # project line
   window = vis.line(X=np.arange(len(data)), Y=data, win=window, update='replace')
   # project images
   window = images(images, padding=5, win=window, nrow=2)


A summary of code 
"""""""""""""""""""""""""

In the following code, I have generated three windows for loss of 'test', 'train', 'recon' and two windows for 'train_image', 'test_image'::

	import visdom
	import numpy as np
	# Start the server in terminal;
	# #  visdom/ python -m visdom.server

	class Visualize_Training():
	    
	    def __init__(self):
		self.vis = visdom.Visdom()
		self.win1 = None
		self.win3 = None
		self.win2 = None
		self.train_images = None
		self.test_images = None


	    
	    def Plot_Progress(self, path, window = "train"):
		'''
		 Plot progress
		'''

		#TODO: Graph these on the same graph dummy!!!!!
		try:
		    data = np.loadtxt(path)
		    if window == "train":
		        if self.win1 == None:
		            self.win1 = self.vis.line(X=np.arange(len(data)), Y=data, opts=dict(xlabel='Historical Epoch',
		                             ylabel='Training Loss',
		                             title='Training Loss',
		                             legend=['Training Loss']))
		        else:
		            self.vis.line(X=np.arange(len(data)), Y=data, win=self.win1, update='replace')

		    elif window == "test":
		        if self.win3 == None:
		            self.win3 = self.vis.line(X=np.arange(len(data)), Y=data, opts=dict(xlabel='Historical Epoch',
		                             ylabel='Testing Loss',
		                             title='Testing Loss',
		                             legend=['Testing Loss']))
		        else:
		            self.vis.line(X=np.arange(len(data)), Y=data, win=self.win3, update='replace') 
		            
		    elif window == "recon":
		        if self.win2 == None:
		            self.win2 = self.vis.line(X=np.arange(len(data)), Y=data, opts=dict(xlabel='Historical Epoch',
		                             ylabel='Recon Loss',
		                             title='Recon Loss',
		                             legend=['Recon Loss']))
		        else:
		            self.vis.line(X=np.arange(len(data)), Y=data, win=self.win2, update='replace')                    
		except:
		    pass
		     
		    
	    def Show_Train_Images(self, images, text='Images'):
		'''
		images: a list of same size images
		'''
		
		if self.train_images == None:

		    self.train_images = self.vis.images(images, padding=5, nrow=2, opts=dict(title=text))
		else:

		    self.vis.images(images, padding=5, win=self.train_images, nrow=2, opts=dict(title=text))

	    def Show_Test_Images(self, images, text='Images'):
		'''
		images: a list of same size images
		'''

		if self.test_images == None:

		    self.test_images = self.vis.images(images, padding=5, nrow=2, opts=dict(title=text))
		else:

		    self.vis.images(images, padding=5, win=self.test_images, nrow=2, opts=dict(title=text))


.. note::
   
   Try to generate your own visdom server!



