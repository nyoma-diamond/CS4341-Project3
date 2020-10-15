from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import random
import numpy as np
from time import time 
from keras import callbacks, regularizers

LABELS_NUMS = [0,1,2,3,4,5,6,7,8,9]

images = np.load("images.npy")

PIXEL_COUNT = len(images[0]) * len(images[0][0])

images = np.ma.array(np.reshape(
				np.load("images.npy"), 
				(len(images), PIXEL_COUNT)
			), 
			mask=False)

labels = to_categorical(np.ma.array(np.load("labels.npy"), mask=False), 10)


# Model
model = Sequential() # declare model
model.add(Dense(512, input_shape=(PIXEL_COUNT, ), kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.001))) # first layer
#model.add(Dropout(0.5))
model.add(Activation('tanh'))

model.add(Dense(128, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dropout(0.3))
model.add(Activation('relu'))

model.add(Dense(32, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.001))) 
#model.add(Dropout(0.2))
model.add(Activation('tanh'))

model.add(Dense(10, kernel_initializer='he_normal')) # output layer
model.add(Activation('softmax'))
print(model.summary())
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True) #TODO: get this working

# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Get training sets
training_images, test_images, training_labels, test_labels = train_test_split(
	images, labels, train_size = 0.75, shuffle = True
)


es_callback = callbacks.EarlyStopping(monitor='val_loss', patience=100)

# Train Model
history = model.fit(training_images, training_labels,
					validation_split=.2,
                    #validation_data = (validation_images, validation_labels), 
                    epochs=30000, 
                    batch_size=512,
					shuffle=True,
					callbacks=[es_callback])

model.save("model")

# Report Results
hist = history.history
plt.plot(range(len(hist.get("accuracy"))), hist.get("accuracy"), label="accuracy")
plt.plot(range(len(hist.get("val_accuracy"))), hist.get("val_accuracy"), label="val_accuracy")
plt.legend(loc="lower right")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("accuracy.png")

plt.clf()

plt.plot(range(len(hist.get("loss"))), hist.get("loss"), label="loss")
plt.plot(range(len(hist.get("val_loss"))), hist.get("val_loss"), label="val_loss")
plt.legend(loc="upper right")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("loss.png")

plt.clf()

predictions = model.predict(test_images)

con_matrix = confusion_matrix([np.argmax(t) for t in test_labels], [np.argmax(p) for p in predictions]) #TODO: axis labels

ax = plt.gca()

plt.imshow(con_matrix, interpolation="nearest")
plt.colorbar()
plt.xticks(LABELS_NUMS)
plt.yticks(LABELS_NUMS)
plt.ylabel("true label")
plt.xlabel("predicted label")

ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

for i in LABELS_NUMS:
	for j in LABELS_NUMS:
		plt.text(j, i, con_matrix[i,j], ha="center", va="center", color="w")

plt.savefig("confusion-matrix.png")
