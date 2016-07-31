import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

batch_size = 128
nb_epoch = 25
is_theano = False # Pooling with border_mode='same' is not possible for Theano.

def get_score(probs, T, label):
	a = np.zeros((probs.shape[0]-T+1, probs.shape[1]))
	for i in range(probs.shape[0]-T+1):
		a[i,:] = np.sum(probs[i:i+T], axis = 0)
	est_label = np.argmax(a, axis = 1)	
	return np.sum(est_label == label), np.sum(est_label != label)

def duration_accuracy(probs, labels):
	ul = np.unique(labels)
	for T in range(1,10):
		n_correct = 0; n_incorrect = 0
		for label in ul:
			idx = np.where(labels == label)[0]
			probs_label = probs[idx]
			n_c, n_ic = get_score(probs_label, T, label)
			n_correct += n_c; n_incorrect += n_ic
		accuracy = n_correct / float(n_correct + n_incorrect)	
		print "For duration %d seconds, the accuracy is %f."%((T+1)*2,accuracy)

def main():
	train_X = np.load('../npy/train_X.npy')
	train_y = np.load('../npy/train_y.npy')-1
	test_X = np.load('../npy/test_X.npy')
	test_y = np.load('../npy/test_y.npy')-1

	train_X = np.transpose(train_X, (0,3,1,2))
	test_X = np.transpose(test_X, (0,3,1,2))

	nb_classes = train_y.max() + 1

	train_Y = np_utils.to_categorical(train_y, nb_classes)
	test_Y = np_utils.to_categorical(test_y, nb_classes)

	model = Sequential()

	model.add(Convolution2D(128, 20, 50, 
	                        border_mode='valid',
	                        input_shape=(2,60, 50)))
	if is_theano:
		model.add(MaxPooling2D(pool_size=(20, 1), strides=(15,1), border_mode='valid'))
	else:
		model.add(MaxPooling2D(pool_size=(20, 1), strides=(15,1), border_mode='same'))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('sigmoid'))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	print model.summary()

	adam = Adam(0.001)
	#adagrad = Adagrad(lr=0.01)
	model.compile(loss='categorical_crossentropy',
	              optimizer=adam,
	              metrics=['accuracy'])

	model.fit(train_X, train_Y, batch_size=batch_size, nb_epoch=nb_epoch,
	          verbose=1, validation_data=(test_X, test_Y))
	score = model.evaluate(test_X, test_Y, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	probs = model.predict(test_X)
	duration_accuracy(probs, test_y)

if __name__ == "__main__":
	main()	
	