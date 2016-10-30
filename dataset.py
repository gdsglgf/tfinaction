import os
import random
from Queue import Queue
from threading import Thread
import time

from PIL import Image
import numpy

def shuffle(data):
	perm = range(len(data))
	random.shuffle(perm)
	shuffled = [data[i] for i in perm]
	return shuffled

def load_image_file(data_dir, join_path=False, image_suffix='.jpg', label=None):
	if label:
		files = [f for f in os.listdir(data_dir) if f.endswith(image_suffix) and f.startswith(label)]
	else:
		files = [f for f in os.listdir(data_dir) if f.endswith(image_suffix)]
	if join_path:
		files = [os.path.join(data_dir, f) for f in files]
	return shuffle(files)

def load_label(data_dir, delimiter='_', image_suffix='.jpg'):
	files = load_image_file(data_dir, False, '.jpg')
	labels = set()
	for f in files:
		label = f.split(delimiter)[0]
		labels.add(label)
	labels = list(labels)
	labels.sort()
	return labels, len(labels)

def dense_to_one_hot(labels_dense, num_classes):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = labels_dense.shape[0]
	index_offset = numpy.arange(num_labels) * num_classes
	labels_one_hot = numpy.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot

# ==============================================================
'''Constants - Feel free to change'''
data_dir = 'test/data'		# image directory
delimiter = '.'
label_names, num_classes = load_label(data_dir, delimiter)
data_rate=[0.8, 0.1, 0.1]	# a list of train, validation, test data rate
# ==============================================================

def extract_label(filename, one_hot=False):
	label = filename.split('/')[-1].split(delimiter)[0]
	label_index = label_names.index(label)
	if one_hot:
		return dense_to_one_hot(numpy.array([label_index]), num_classes)[0]
	return label_index

def load_files_by_class(data_dir, delimiter):
	print('data_dir:%s, delimiter:%s' %(data_dir, delimiter))
	label_names, num_classes = load_label(data_dir, delimiter)
	files = [load_image_file(data_dir, join_path=True, label=l) for l in label_names]
	counter = [len(f) for f in files]

	return files, counter

def random_build(data_dir=data_dir, delimiter=delimiter):
	files, counter =  load_files_by_class(data_dir, delimiter)
	
	dataset = [[], [], []]
	for i, rate in enumerate(data_rate, start=0):
		for j, f in enumerate(files, start=0):
			random.shuffle(f)
			end = int(rate * counter[j])
			dataset[i].extend(f[: end])
	train, validation, test = dataset
	random.shuffle(train)
	random.shuffle(validation)
	random.shuffle(test)

	print(len(train), len(validation), len(test))
	return train, validation, test

def split_build(data_dir=data_dir, delimiter=delimiter):
	files, counter =  load_files_by_class(data_dir, delimiter)

	train, validation, test = [], [], []
	for i, f in enumerate(files, start=0):
		random.shuffle(f)
		start = int(data_rate[0] * counter[i])
		train.extend(f[: start])
		end = counter[i] - int(data_rate[2] * counter[i])
		validation.extend(f[start: end])
		test.extend(f[end: ])
		# print('start:%d, end:%d' %(start, end))

	random.shuffle(train)
	random.shuffle(validation)
	random.shuffle(test)
	print(len(train), len(validation), len(test))
	return train, validation, test

def read_image(files, num_worker_threads=5, size=(227, 227), one_hot=False):
	input_queue = Queue()
	for f in files:
		input_queue.put(f)
	# print('input queue size:%d' %(input_queue.qsize()))
	output_queue = Queue()
	def worker():
		while not input_queue.empty():
			filename = input_queue.get()
			image = numpy.array(Image.open(filename).convert('RGB').resize(size))
			# Convert from [0, 255] -> [0.0, 1.0]
			image = image.astype(numpy.float32)
			image = numpy.multiply(image, 1.0 / 255.0)

			label = extract_label(filename, one_hot=one_hot)
			output_queue.put((image, label))
			input_queue.task_done()
	for i in range(num_worker_threads): # start threads
		worker_thread = Thread(target=worker)
		worker_thread.daemon = True
		worker_thread.start()
	input_queue.join() # block until all tasks are done

	images = []
	labels = []
	while not output_queue.empty():
		image, label = output_queue.get()
		images.append(image)
		labels.append(label)

	return numpy.array(images), numpy.array(labels)

class Dataset(object):
	"""docstring for Dataset"""
	def __init__(self, files):
		super(Dataset, self).__init__()
		self._files = shuffle(files)

		self._num_examples = len(files)
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def files():
		return self._files

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def num_epochs(self, batch_size):
		"""Return the total epochs in this data set by given batch_size."""
		return self._num_examples // batch_size

	def next_batch_file(self, batch_size):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			self._files = shuffle(self._files)
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._files[start:end]
	
	def next_batch(self, batch_size, one_hot=True):
		"""Return the next `batch_size` examples from this data set."""
		batch_file = self.next_batch_file(batch_size)
		return read_image(self.next_batch_file(batch_size), one_hot=one_hot)

	def load_all(self, one_hot=True):
		return read_image(self._files, one_hot=one_hot)

class DatasetQueue(Thread):
	"""docstring for DatasetQueue"""
	def __init__(self, dataset, batch_size=128, qsize=10):
		super(DatasetQueue, self).__init__()
		self._dataset = dataset
		self._batch_size = batch_size
		self._queue = Queue(qsize)
		self._thread_stop = False

	def run(self):
		while not self._thread_stop:
			print('queue size:%d' %(self._queue.qsize()))
			images, labels = self._dataset.next_batch(self._batch_size)
			self._queue.put((images, labels))

	def stop(self):
		self._thread_stop = True

	def next_batch(self):
		images, labels = self._queue.get()
		return images, labels