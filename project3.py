from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.vis_utils import plot_model
from keras.utils import plot_model, to_categorical
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import math
import numpy as np
from time import time 
from keras import callbacks, regularizers

LABELS_NUMS = [0,1,2,3,4,5,6,7,8,9]

images = np.load("images.npy")

PIXEL_COUNT = len(images[0]) * len(images[0][0])

images = np.reshape(images, (len(images), PIXEL_COUNT))

labels = to_categorical(np.ma.array(np.load("labels.npy"), mask=False), 10)


# Model
model = Sequential() # declare model
model.add(Dense(512, input_shape=(PIXEL_COUNT, ), kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.001))) # first layer
#model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Dense(128, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.001)))
#model.add(Dropout(0.3))
model.add(Activation('relu'))

model.add(Dense(32, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.001))) 
#model.add(Dropout(0.2))
model.add(Activation('tanh'))

model.add(Dense(10, kernel_initializer='he_normal')) # output layer
model.add(Activation('softmax'))
print(model.summary())

try:
	plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True) #TODO: get this working
except:
	print("Model plot could not be generated.")

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
                    epochs=200, 
                    batch_size=512,
					shuffle=True,
					callbacks=[es_callback])

model.save("model.tf")

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

plt.clf()

print("Plotting incorrect predictions...")

unflattened = np.reshape(test_images, (len(test_images), round(math.sqrt(PIXEL_COUNT)), round(math.sqrt(PIXEL_COUNT))))

count = 0

for i in range(len(predictions)):
	if np.argmax(predictions[i]) != np.argmax(test_labels[i]):
		# print("P: ", end="")
		# print(np.argmax(predictions[i]), end=" | ")
		# print("A: ", end="")
		# print(np.argmax(test_labels[i]))
		count += 1

fig = plt.figure(figsize=(round(math.sqrt(count)), round(math.sqrt(count))))
fig.suptitle("Incorrect predictions and their corresponding images", fontsize=20)

i = 0
j = 1
while i < len(unflattened) and j < round(math.sqrt(count))*round(math.sqrt(count)):
	if np.argmax(predictions[i]) != np.argmax(test_labels[i]):
		# print("P: ", end="")
		# print(np.argmax(predictions[i]), end=" | ")
		# print("A: ", end="")
		# print(np.argmax(test_labels[i]))

		fig.add_subplot(round(math.sqrt(count)), round(math.sqrt(count)), j)
		j += 1
		ax = plt.gca()
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		plt.imshow(unflattened[i], cmap='gray')
		plt.title(np.argmax(predictions[i]))
	i += 1

plt.tight_layout()

plt.savefig("incorrect-predictions.png")
