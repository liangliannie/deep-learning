.. badblock documentation master file, created by
   sphinx-quickstart on Thu Mar  7 09:42:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Prepare your data
====================================

Process the raw data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Read and reshape
"""""""""""""""""""
The size(shape) and the distribution of the data would affect both the performance of the output and the learning speed of the network,
and hence pre-process the raw data to the shape/distribution we want and post-processing it back to the origin format are usually common
in machine learning[The reasons behind this are a lot, say we want the learning converge similar to all directions in our training data]. 
Mostly, the methods include but not limited to normalization, reshaping, etc.

In our case, we have .s file as both the input file and target file with both files around 1.4G(with flatten data stored as uint16), and our goal is to using U-Net to learn
the data in the input file and to make it close to the data in the target file. In fact, these files are too large as one input for neural network, because I do not want to load all the datas into the memory if we do not have a lot memory. (It is so important! Hence, I will take this three times: We do not want to feed all into the memory!We do not want to feed all into the memory!We do not want to feed all into the memory!).

To advoid feed all the data into the memory, we do reshape the data to more informative matrix and partition the data into smaller pieces, and then we could let the network learn smaller pieces with the pain that we will lose the relations among the smaller pieces. In my cases, I am working on sinograms and the sinograms have their own informative structure, and then I used [TOF, Slice, W, L] as my shape of the data, where I feed the network based on slices. 

Here, for each slice, the image has shape [50, 520]. For most cases in machine learning without down/up sampling in the images, the number of W and L does not matter. But since we are working on UNet (as autoencoder, we need to encoder and then decode the data) we would better make W and L as a power of 2 (since we will consider the network structure as UNet which will include both downsampling and upsampling).

.. note::
   An example of reshaping to a power of 2:

	`Numpy <http://www.numpy.org/>`_ provides the padding function::

	    data = np.pad(data, ((x1, y1), (x2, y2), (x3, y3), (x4, y4)), 'wrap') 

	In our case, if we have [TOF=34, Slice=815, W=50, L=520], we can do::

	    data = np.pad(data, ((0, 0), (0, 0), (7, 7), (0, 0)), 'wrap')

	to make it [TOF=34, Slice=815, W=64, L=520], which would be good enough for 3 times downsampling since W and L can be divided by :math:`2^3`.

   Here I only list one way to change the shape, and actually there might be tons of other methods which are good to try.

Save the data in pickle
"""""""""""""""""""""""""""""

`Pickle <https://docs.python.org/3/library/pickle.html>`_ module implement binary protocols for serializing and de-serializing a Python object structure. Here we could just dump our matrix into pickle by using::

    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
        
.. note::
   Be careful with you pickle version, since it might not match between python2 and python3.

The details of the data process can be refered from sino_process_tof.py

.. toctree::
   :maxdepth: 2
      
   /usage/tof_file.rst


.. warning::
   
   Loading all the data into the GPU is not only time consuming and not efficient.

Load your data 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dataset Class
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
The abstract class in pytorch **torch.utils.data.Dataset** is the main class to call for loading data, and there are mainly two methods would be called.

* __len__: which returns the size/length of the dataset
* __getitem__: which returns one sample of the dataset based on the index, such that dataset[i] for index i

In our case, we define the class :code:`class Sino_Dataset(dataset)` inherits from :code:`torch.utils.data.Dataset`

Here, the length of the dataset is::

    def __len__(self):

        return self.epoch_size

and the item of the dataset is::

   def __getitem__(self, idx):
	
	return self.data[idx]

the :code:`self.data.shape=[tof*slice, W, L]`

The details of getting item also include shuffling and file updating, which can be viewed as different methods to improve the randomness of the data set.

.. toctree::
   :maxdepth: 2
      
   /usage/datset_file.rst











