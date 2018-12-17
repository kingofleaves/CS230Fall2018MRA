from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from skimage import measure
import numpy as np
import util
import pydicom
import os
import numpy as np
from matplotlib import pyplot, cm
import glob
from PIL import Image


fileDir = '/data2/yeom/ky_mra'
pattern_match = '/**/**/**/*MRA*/*.dcm'
normals = glob.glob(fileDir + '/Normal_MRA' + pattern_match)
abnormals = glob.glob(fileDir + '/MMD_MRA' + pattern_match)

print(len(normals))
print(len(abnormals))

### Returns matrix of [m, n], where m is the number of patients, and m is the number of features from the flattened image matrix ###
def pull_condensed_matrix(curr_patient, dir_list) :
	flat_shape = 64*64*3
	patient_mat = []
	return_mat = []
	for filepath in dir_list:
		dir_name = filepath[:filepath.rfind('/')]
		if curr_patient != dir_name and patient_mat != [] :
			curr_patient = dir_name
			patient_mat = np.asarray(patient_mat)
			if len(patient_mat) != 0 and patient_mat.shape[1] == flat_shape :
				collapsed_img = np.mean(patient_mat, axis = 0)
				print('Matrix to be appended for a new patient: ' + str(collapsed_img))
				return_mat.append(collapsed_img)
				print('Shape of collapsed_img for patient ' + str(curr_patient) + ' : ' + str(collapsed_img.shape))
			patient_mat = []
			curr_patient = dir_name
		data = pydicom.dcmread(filepath)
		try:
			dim = 64
			data_np = data.pixel_array
			pool_size = int(data_np.shape[0]/dim)
			data_np = measure.block_reduce(data_np, (pool_size,pool_size), np.mean)
			data_np = np.repeat(data_np[:, :, np.newaxis], 3, axis=2)
			mat_to_append = data_np.flatten()
			if mat_to_append.shape[0] == flat_shape : 
				patient_mat.append(mat_to_append)
		except Exception:
			pass
	collapsed_img = np.mean(patient_mat, axis = 0)
	return_mat.append(collapsed_img)
	return_mat = np.ma.row_stack(return_mat)
	print(return_mat.shape)
	return(return_mat) # Shape is (m, 256, 256)

last_dir = normals[0].rfind('/')
first_normal = normals[0][:last_dir]
print('Pulling normals matrix...')
normals_mat = pull_condensed_matrix(first_normal, normals)

last_dir = abnormals[0].rfind('/')
first_abnormal = abnormals[0][:last_dir]
print('Pulling abnormal matrix...')
abnormals_mat = pull_condensed_matrix(first_abnormal, abnormals)



mixed_mat = np.concatenate((normals_mat,abnormals_mat),axis = 0)
mixed_labels = np.ones((normals_mat.shape[0]+abnormals_mat.shape[0], ))
mixed_labels[:normals_mat.shape[0]] = 0
print('Shape of the data: ' + str(mixed_mat.shape))
print('Shape of the labels: ' + str(mixed_labels.shape))



train_img, test_img, train_label, test_label = train_test_split(
	mixed_mat, mixed_labels, test_size=1/7.0, random_state=0)
print('Shape of train_img: ' + str(train_img.shape))
print('Shape of test_img: ' + str(test_img.shape))
print('Shape of train_label: ' + str(train_label.shape))
print('Shape of test_label: ' + str(test_label.shape))

logisticRegr = LogisticRegression(solver = 'lbfgs')

print('Training Logistic Regression...')
logisticRegr.fit(train_img, train_label)

train_predictions = logisticRegr.predict(train_img)
test_predictions = logisticRegr.predict(test_img)


train_acc = np.sum(train_predictions == train_label) / float(len(train_label))
train_recall = np.sum(train_label[(train_predictions == 1)])/(np.sum(train_label[(train_predictions == 1)]) + np.sum(train_label[(train_predictions == 0)]))
train_precision = np.sum(train_label[(train_predictions == 1)])/np.sum(train_predictions)
train_f1_score = 2*train_precision*train_recall/(train_precision+train_recall)

test_acc = np.sum(test_predictions == test_label) / float(len(test_label))
test_recall = np.sum(test_label[(test_predictions == 1)])/(np.sum(test_label[(test_predictions == 1)]) + np.sum(test_label[(test_predictions == 0)]))
test_precision = np.sum(test_label[(test_predictions == 1)])/np.sum(test_predictions)
test_f1_score = 2*test_precision*test_recall/(test_precision+test_recall)

print('Train metrics: ' )
print('Accuracy: ' + str(train_acc))
print('F1 score: ' + str(train_f1_score))
print('Percent of predictions that were correct: ' + str(train_precision))
print('Test metrics: ')
print('Accuracy: ' + str(test_acc))
print('F1 Score: ' + str(test_f1_score))
print('Percent of predictions that were correct: ' + str(test_precision))

np.savetxt('./LR_test_predictions2.txt', test_predictions)
np.savetxt('./LR_test_labels2.txt', test_label)
np.savetxt('all_data_matrix.txt', mixed_mat, fmt='%f')
np.savetxt('all_labels.txt', mixed_labels, fmt='%f')    


