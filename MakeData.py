from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import cv2
import re
import random

class myAugmentation(object):

	def __init__(self, train_path="train", label_path="label", merge_path="merge", aug_merge_path="aug_merge", aug_train_path="aug_train", aug_label_path="aug_label", img_type="tif"):
		"""
		使用glob从路径中得到所有的“.img_type”文件，初始化类：__init__()
		"""
		self.train_imgs = glob.glob(train_path+"/*."+img_type)
		self.label_imgs = glob.glob(label_path+"/*."+img_type)
		self.train_path = train_path
		self.label_path = label_path
		self.merge_path = merge_path
		self.img_type = img_type
		self.aug_merge_path = aug_merge_path
		self.aug_train_path = aug_train_path
		self.aug_label_path = aug_label_path
		self.slices = len(self.train_imgs)


	def Augmentation(self):
		"""
		Start augmentation.....
		"""
		trains = self.train_imgs
		labels = self.label_imgs
		path_train = self.train_path
		path_label = self.label_path
		path_merge = self.merge_path
		imgtype = self.img_type
		path_aug_merge = self.aug_merge_path

		if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
			print ("trains can't match labels")
			return 0
		for file_path in trains:
			(filepath, tempfilename) = os.path.split(file_path)
			(imName, extension) = os.path.splitext(tempfilename)
			img_t = cv2.cvtColor(cv2.imread(path_train+"/"+imName+"."+imgtype), cv2.COLOR_BGR2GRAY)
			img_l = cv2.cvtColor(cv2.imread(path_label+"/"+imName+"."+imgtype), cv2.COLOR_BGR2GRAY)
			imgH = img_t.shape[0]
			imgW = img_t.shape[1]
			cutH = 512
			cutW = 512
			count = 1
			if(imgH<cutH or imgW<cutW):
				if imgH<imgW:
					imgW = (512/imgH)*imgW
					imgH = 512
				else:
					imgH = (512/imgW)*imgH
					imgW = 512
				img_t = cv2.resize(img_t, (int(imgW),int(imgH)))
				img_l = cv2.resize(img_l, (int(imgW),int(imgH)))
			while 1:
				# 随机产生x,y   此为像素内范围产生
				y = random.randint(0, int(imgH-cutH))
				x = random.randint(0, int(imgW-cutW))
				# 随机截图
				cropImg = img_t[(y):(y + cutH), (x):(x + cutW)]
				cropLaber = img_l[(y):(y + cutH), (x):(x + cutW)]
				if np.sum(cropLaber) >1000000 :
					cv2.imwrite(aug_train_path + '/' + imName + '_' + str(count) + '.tif', cropImg)
					cv2.imwrite(aug_label_path + '/' + imName + '_' + str(count) + '.tif', cropLaber)
					count += 1
				if count == 40:
					break




			# x_t = img_to_array(img_t)
			# x_l = img_to_array(img_l)
			#
			# imgMerge = np.ndarray((out_rows, out_cols, 2), dtype=np.uint8)
			# imgMerge[:,:,0] = x_t
			# imgMerge[:,:,1] = x_l
			# img_tmp = array_to_img(imgMerge)
			# img_tmp.save(path_merge+"/"+imName+"."+imgtype)
			#
			# img = imgMerge
			# img = img.reshape((1,) + img.shape)
			# savedir = path_aug_merge + "/" + str(i)
			# if not os.path.lexists(savedir):
			# 	os.mkdir(savedir)
			# self.doAugmentate(img, savedir, str(i))

	def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):
		# 增强一张图片的方法
		"""
		augmentate one image
		"""
		datagen = self.datagen
		i = 0
		for batch in datagen.flow(img,
						  batch_size=batch_size,
						  save_to_dir=save_to_dir,
						  save_prefix=save_prefix,
						  save_format=save_format):
			i += 1
			if i > imgnum:
				break

	def splitMerge(self):
		# 将合在一起的图片分开
		"""
		split merged image apart
		"""
		path_merge = self.aug_merge_path
		path_train = self.aug_train_path
		path_label = self.aug_label_path

		for i in range(self.slices):
			path = path_merge + "/" + str(i)
			train_imgs = glob.glob(path+"/*."+self.img_type)
			savedir = path_train + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			savedir = path_label + "/" + str(i)

			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			for imgname in train_imgs:
				midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
				img = cv2.imread(imgname)
				img_train = img[:,:,2] #cv2 read image rgb->bgr
				img_label = img[:,:,0]
				cv2.imwrite(path_train+"/"+str(i)+"/"+midname+"_train"+"."+self.img_type,img_train)
				cv2.imwrite(path_label+"/"+str(i)+"/"+midname+"_label"+"."+self.img_type,img_label)

	def splitTransform(self):
		# 拆分透视变换后的图像
		"""
		split perspective transform images
		"""
		#path_merge = "transform"
		#path_train = "transform/data/"
		#path_label = "transform/label/"

		path_merge = "deform/deform_norm2"
		path_train = "deform/train/"
		path_label = "deform/label/"

		train_imgs = glob.glob(path_merge+"/*."+self.img_type)
		for imgname in train_imgs:
			midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
			img = cv2.imread(imgname)
			img_train = img[:,:,2]#cv2 read image rgb->bgr
			img_label = img[:,:,0]
			cv2.imwrite(path_train+midname+"."+self.img_type,img_train)
			cv2.imwrite(path_label+midname+"."+self.img_type,img_label)

class dataProcess(object):
	def __init__(self, out_rows, out_cols, data_path = "./images/train/images", label_path = "./images/train/label", test_path = "./images/test", npy_path = "./images/train/npydata", img_type = "tif"):
		# 数据处理类，初始化
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path

# 创建训练数据
	def create_train_data(self):
		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		print(len(imgs))

		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			img = cv2.imread(self.data_path + "/" + midname)
			label = cv2.imread(self.label_path + "/" + midname)
			# img = load_img(self.data_path + "/" + midname,grayscale = True)
			# label = load_img(self.label_path + "/" + midname,grayscale = True)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
			img = img_to_array(img)
			label = img_to_array(label)
			#img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			#label = np.array([label])
			imgdatas[i] = img
			imglabels[i] = label
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to .npy files done.')

# 创建测试数据
	def create_test_data(self):
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		imgnames = []
		for imgname in imgs:
			midname = imgname[imgname.rindex("/")+1:]
			# img = load_img(self.test_path + "/" + midname,grayscale = True)
			img = cv2.imread(self.test_path + "/" + midname)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			img = img_to_array(img)
			#img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			imgdatas[i] = img
			imgnames.append(midname)
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		y1 = np.array(imgnames)
		y1 = y1.reshape(len(imgnames), 1)
		np.save(self.npy_path + '/imgs_name.npy', y1)
		print('Saving to imgs_test.npy files done.')

# 加载训练图片与mask
	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		mean = imgs_train.mean(axis = 0)
		imgs_train -= mean
		imgs_mask_train /= 255
		# 做一个阈值处理，输出的概率值大于0.5的就认为是对象，否则认为是背景
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train,imgs_mask_train

# 加载测试图片
	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		mean = imgs_test.mean(axis = 0)
		imgs_test -= mean
		return imgs_test

	def load_test_name(self):
		imgs_test_name = np.load(self.npy_path+"/imgs_name.npy")
		return imgs_test_name

if __name__ == "__main__":
	train_path = "/media/we/work/TrainData/SpeechBalloon/image"
	label_path = "/media/we/work/TrainData/SpeechBalloon/label"
	merge_path = "/media/we/work/TrainData/SpeechBalloon/merge"
	aug_merge_path = "/media/we/work/TrainData/SpeechBalloon/aug_merge"
	aug_train_path = "/media/we/work/TrainData/SpeechBalloon/aug_image"
	aug_label_path = "/media/we/work/TrainData/SpeechBalloon/aug_label"
	img_type = "jpg"
	aug = myAugmentation(train_path, label_path, merge_path, aug_merge_path, aug_train_path, aug_label_path, img_type)
	aug.Augmentation()
