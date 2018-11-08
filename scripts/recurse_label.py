import pydicom
import os
import numpy as np
from matplotlib import pyplot, cm
import glob
from skimage import measure
import train_resnet_keras

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

fileDir = '/data2/yeom/ky_mra'
pattern_match = '/**/**/**/**/*.dcm'

# End up with two arrays of all normal file names and all abnormal file names
normals = glob.glob(fileDir + '/Normal_MRA' + pattern_match)
abnormals = glob.glob(fileDir + '/MMD_MRA' + pattern_match)

print(len(normals))
print(len(abnormals))

normals_mat = []
for filepath in normals[:1000]:
#  print('Processing %s' % filepath)
  data = pydicom.dcmread(filepath)
  data_np = data.pixel_array
  pool_size = int(data_np.shape[0]/64)
  data_np = measure.block_reduce(data_np, (pool_size,pool_size), np.mean)
  data_np = np.repeat(data_np[:, :, np.newaxis], 3, axis=2)
  normals_mat.append(data_np)
normals_mat = np.asarray(normals_mat) # Shape is (m, 256, 256)

abnormals_mat = []
for filepath in abnormals[:1000]:
#  print('Processing %s' % filepath)
  data = pydicom.dcmread(filepath)
  data_np = data.pixel_array
  pool_size = int(data_np.shape[0]/64)
  data_np = measure.block_reduce(data_np, (pool_size,pool_size), np.mean)
  data_np = np.repeat(data_np[:, :, np.newaxis], 3, axis=2)
  abnormals_mat.append(data_np)
abnormals_mat = np.asarray(abnormals_mat) # Shape is (m, 256, 256)

normals_labels = np.zeros((normals_mat.shape[0], 6)) # Shape is (m, 6)
normals_labels[:,0] = 1
abnormals_labels = np.zeros((abnormals_mat.shape[0], 6)) # Shape is (m, 6)
abnormals_labels[:5] = 1

print(abnormals_mat.shape)
# print(normals[0])
print(normals_mat.shape)
print(normals_labels.shape)

print(np.max(normals_mat))
print(np.max(abnormals_mat))

# pyplot.figure()
# pyplot.imshow(normals_mat[0])
# pyplot.show()

mixed_mat = np.concatenate((normals_mat,abnormals_mat),axis = 0)
mixed_labels = np.concatenate((normals_labels,abnormals_labels),axis = 0)

indexes = np.arange(mixed_mat.shape[0])
np.random.shuffle(indexes)

shuffled_mat = mixed_mat[indexes]
shuffled_labels = mixed_labels[indexes]

end_train_index = int(len(shuffled_mat) * 0.95)
X_train = shuffled_mat[0:end_train_index]
print(X_train.shape)
Y_train = shuffled_labels[0:end_train_index]
print(Y_train.shape)

X_test = shuffled_mat[end_train_index:]
print(X_test.shape)
Y_test = shuffled_labels[end_train_index:]
print(Y_test.shape)
 
train_resnet_keras.train_resnet(X_train, Y_train, X_test, Y_test)
