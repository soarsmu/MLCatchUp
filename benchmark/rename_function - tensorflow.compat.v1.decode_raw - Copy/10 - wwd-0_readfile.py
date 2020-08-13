import  tensorflow as tf
import  tensorflow.compat.v1 as tfv1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tensorflow 读取数据学习  -- 读取csv文件
tfv1.disable_eager_execution()

# 定义cifar的数据等命令行参数
FLAGS = tfv1.app.flags.FLAGS
tfv1.app.flags.DEFINE_string("cifar_dir","./binary","文件的目录")

class CifarRead(object):
	"""
	完成读取二进制文件 写进tfrecords 读取rfrecords
	"""
	def __init__(self,filelist):
		# 文件列表
		self.file_list = filelist
		# 定义读取的图片的属性
		self.height = 32
		self.width = 32
		self.channel = 3

		# 二进制文件每张图片的字节
		self.label_bytes = 1
		self.image_bytes = self.height * self.width * self.channel
		self.bytes = self.label_bytes + self.image_bytes


	def read_and_decode(self):
		# 构造文件队列
		file_queue = tfv1.train.string_input_producer(self.file_list)

		# 构造二进制文件读取器，读取内容
		reader = tfv1.FixedLengthRecordReader(self.bytes)
		key,value = reader.read(file_queue)

		# 解码内容 二进制内容的解码
		label_image = tfv1.decode_raw(value,tf.uint8)

		# 分割图片和标签数据，特征值和目标值
		label = tf.cast(tf.slice(label_image,[0],[self.label_bytes]),tf.int32)

		image = tf.slice(label_image, [self.label_bytes], [self.image_bytes])

		# 可以对图片的特征数据进行形状的改变 [3072] -> [32,32,3]
		image_reshape = tf.reshape(image,[self.height,self.width,self.channel])

		# 批处理数据
		image_batch,label_batch = tfv1.train.batch([image_reshape,label],batch_size=10,num_threads=1,capacity = 10)
		print(image_batch,label_batch)

		return image_batch,label_batch

	def bainaryread(filelist):

		return None


def csvread(filelist):
	"""
	读取csv文件
	:param filelist: 文件列表 + 名字的列表
	:return: 读取的内容
	"""

	# 1、构造文件队列
	file_queue = tfv1.train.string_input_producer(filelist)

	# 2、构造csv阅读器读取队列数据
	reader = tfv1.TextLineReader()
	key ,value = reader.read(file_queue)

	# 3、对每行数据进行解码
	# record_defaults:指定每一个样本的每一列的类型，指定默认值
	records = [["None"],["None"]]

	example,lable = tfv1.decode_csv(value,record_defaults=records )

	# 4、想要读多个数据，就需要批处理
	example_batch,lable_batch = tfv1.train.batch([example,lable],batch_size=4,num_threads=1,capacity=5)

	return example_batch,lable_batch

if __name__ =="__main__":
	# 1、找到文件，放入列表 路径+名字 -》列表
	file_name = os.listdir(FLAGS.cifar_dir)

	filelist = [os.path.join(FLAGS.cifar_dir,file) for file in file_name if file[-3:] == "bin"]

	cf = CifarRead(filelist)
	image_batch,lable_batch = cf.read_and_decode()

	#example,lable =
	# (filelist)
	#image_resize = imageread(filelist)
	# 开启会话运行
	with   tfv1.Session() as sess:
		# 定义一个线程协调器
		coord = tf.train.Coordinator()

		# 开启读取文件的线程
		threads = tfv1.train.start_queue_runners(sess,coord=coord)

		print(sess.run([image_batch,lable_batch]))

		coord.request_stop()

		coord.join(threads)
