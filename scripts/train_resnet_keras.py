from resnet_keras_functions import *
import tensorflow as tf
import keras.backend as K


def train_resnet(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig):

  K.set_image_data_format('channels_last')
  K.set_learning_phase(1)

  model = ResNet50(input_shape = (64, 64, 3), classes = 6)
  # model = multi_gpu_model(model, gpus=4)

  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  # X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

  # Normalize image vectors
  X_train = X_train_orig/8192.
  X_test = X_test_orig/8192.

  # Convert training and test labels to one hot matrices
  # Y_train = convert_to_one_hot(Y_train_orig, 6).T
  # Y_test = convert_to_one_hot(Y_test_orig, 6).T

  Y_train = Y_train_orig
  Y_test = Y_test_orig

  print ("number of training examples = " + str(X_train.shape[0]))
  print ("number of test examples = " + str(X_test.shape[0]))
  print ("X_train shape: " + str(X_train.shape))
  print ("Y_train shape: " + str(Y_train.shape))
  print ("X_test shape: " + str(X_test.shape))
  print ("Y_test shape: " + str(Y_test.shape))

  model.fit(X_train, Y_train, epochs = 20, batch_size = 32)

  preds = model.evaluate(X_test, Y_test)
  print ("Loss = " + str(preds[0]))
  print ("Test Accuracy = " + str(preds[1]))
