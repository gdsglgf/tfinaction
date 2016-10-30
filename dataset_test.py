from  dataset import *

small_test_data_dir = 'test/data'

large_test_data_dir = '../../python/datasets/dogs-vs-cats/train'

def show_header(header):
	lines = '-' * 25
	print('%s[%s]%s' %(lines, header, lines))

def test_shuffle():
	show_header('test_shuffle')
	data = range(10)
	print(shuffle(data))
	print(data)

	data = 'abcdefghijk'
	print(shuffle(data))
	print(data)

	data = list('helloworld')
	print(data)
	random.shuffle(data)
	print(data)

def test_load_image_file():
	show_header('test_load_image_file')
	files = load_image_file(small_test_data_dir)
	print(len(files) == 2)
	print(files)

	files = load_image_file(small_test_data_dir, join_path=True)
	print(len(files) == 2)
	print(files)

	files = load_image_file(small_test_data_dir, image_suffix='.png')
	print(len(files) == 0)

	files = load_image_file(small_test_data_dir, label='notexists')
	print(len(files) == 0)

def test_load_label():
	show_header('test_load_label')
	labels = load_label(small_test_data_dir, '.')
	print(len(labels) == 2)
	print(labels)

	labels = load_label(small_test_data_dir, '_')
	print(len(labels) == 0)

def test_dense_to_one_hot():
	show_header('test_dense_to_one_hot')
	labels_dense = numpy.array([0, 1, 2, 0, 1])
	labels_one_hot = dense_to_one_hot(labels_dense, 3)
	print(labels_one_hot)

	labels_one_hot = dense_to_one_hot(numpy.array([0]), 10)
	print(labels_one_hot)
	print(labels_one_hot[0])

def show_constans():
	show_header('show_constans')
	print('data_dir:%s' %(data_dir))
	print('label_delimiter:%s' %(delimiter))
	print('label_names:%s' %(label_names))
	print('num_classes:%d' %(num_classes))
	print('data_rate:%s' %(data_rate))

def test_extract_label():
	show_header('test_extract_label')
	filenames = ['data/cat.10.jpg', 'data/dog.20.jpg']
	for filename in filenames:
		print('filename:%s' %(filename))
		label = extract_label(filename)
		print(label)

		label_one_hot = extract_label(filename, one_hot=True)
		print(label_one_hot)

def test_build_dataset():
	show_header('random_build')
	train, validation, test = random_build(small_test_data_dir)
	train, validation, test = random_build(large_test_data_dir)

	show_header('split_build')
	train, validation, test = split_build(small_test_data_dir)
	train, validation, test = split_build(large_test_data_dir)

def test_read_image():
	show_header('test_read_image')
	files = load_image_file(small_test_data_dir, join_path=True)
	images, labels = read_image(files)
	print images.shape, labels.shape

	images, labels = read_image(files, one_hot=True)
	print images.shape, labels.shape

def test_dataset():
	show_header('mock_dataset')
	mock_files = range(1000)
	num_mock_files = len(mock_files)
	mock_dataset = Dataset(mock_files)
	print(mock_dataset.num_examples == num_mock_files)
	print(mock_dataset.num_epochs(128) == (num_mock_files // 128))
	print(mock_dataset.next_batch_file(10))
	print(mock_dataset.next_batch_file(10))

	show_header('small_dataset')
	small_files = load_image_file(small_test_data_dir, join_path=True)
	small_dataset = Dataset(small_files)
	print(small_dataset.num_examples)
	images, labels = small_dataset.load_all(one_hot=False)
	print images.shape, labels.shape

	images, labels = small_dataset.load_all(one_hot=True)
	print images.shape, labels.shape

	show_header('large_dataset')
	large_files = load_image_file(large_test_data_dir, join_path=True)
	large_dataset = Dataset(large_files)
	print(large_dataset.num_examples)
	images, labels = large_dataset.next_batch(128, one_hot=True)
	print images.shape, labels.shape

	images, labels = large_dataset.next_batch(64, one_hot=True)
	print images.shape, labels.shape

def test_datasetqueue():
	show_header('test_datasetqueue')
	large_files = load_image_file(large_test_data_dir, join_path=True)
	large_dataset = Dataset(large_files)
	queue = DatasetQueue(large_dataset, batch_size=128)
	queue.start()

	for i in range(50):
		images, labels = queue.next_batch()
		print i, images.shape, labels.shape
		time.sleep(0.01)

	queue.stop()

if __name__ == '__main__':
	# test_shuffle()
	# test_load_image_file()
	# test_load_label()
	# test_dense_to_one_hot()
	# show_constans()
	# test_extract_label()
	# test_build_dataset()
	# test_read_image()
	# test_dataset()
	test_datasetqueue()
	