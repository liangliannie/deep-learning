.. badblock documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:42:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Visualize your results!
====================================


Install Visdom
^^^^^^^^^^^^^^^^^
Visdom is a flexible tool for creating, organizing, and sharing visualizations of live, rich data. Supports Torch and Numpy, which can be found `<https://github.com/facebookresearch/visdom/>`_.

Use Visdom
^^^^^^^^^^^^^^^^^
Try to view your output in websites::

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


.. toctree::
   :maxdepth: 2
   :caption: Contents:



