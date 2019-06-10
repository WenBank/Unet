from unet import *
from data import *

mydata = dataProcess(512,512)
# imgs_test = mydata.load_test_data()
############################################
img = cv2.imread('/home/we/Pictures/test/3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img_to_array(img)
imgdatas = np.ndarray((1, 512, 512,1), dtype=np.uint8)
imgdatas[0] = img
# np.save("./1.npy",imgdatas)
# imgs_test = np.load("./1.npy")
imgdatas = imgdatas.astype('float32')
imgdatas /= 255
imgs_train,imgs_mask_train,mean = mydata.load_train_data()
# mean = imgdatas.mean(axis=0)
imgdatas -= mean
#############################################
myunet = myUnet()
model = myunet.get_unet()
model.load_weights('my_unet.hdf5')
imgs_mask_test = model.predict(imgdatas, batch_size=1, verbose=1)
# np.save('imgs_mask_test.npy', imgs_mask_test)
for i in range(imgs_mask_test.shape[0]):
    img = imgs_mask_test[i]
    img = array_to_img(img)
    img.save("./%d.jpg" % (i))

# ---------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# imgs_test = np.load('../npydata/imgs_test.npy')
# imgs_test_predict = np.load('../results/imgs_mask_test.npy')
# print(imgs_test.shape, imgs_test_predict.shape)
#
#
# n = 2
# plt.figure(figsize=(20, 4))
# for i in range(20, 22):
#     plt.gray()
#     ax = plt.subplot(2, n, (i-20)+1)
#     plt.imshow(imgs_test[i].reshape(512, 512))
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     ax = plt.subplot(2, n, (i - 20) + n + 1)
#     plt.imshow(imgs_test_predict[i].reshape(512, 512))
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

print("array to image")
imgs = np.load('../results/imgs_mask_test.npy')
for i in range(imgs.shape[0]):
    img = imgs[i]
    img = array_to_img(img)
    img.save("../results/%d.jpg" % (i))