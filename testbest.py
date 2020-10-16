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


model = load_model("best-tested-model.tf")


predictions = model.predict(images)

con_matrix = confusion_matrix([np.argmax(t) for t in labels], [np.argmax(p) for p in predictions]) #TODO: axis labels

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

plt.savefig("full-confusion-matrix.png")

plt.clf()

print("Plotting incorrect predictions...")

unflattened = np.reshape(images, (len(images), round(math.sqrt(PIXEL_COUNT)), round(math.sqrt(PIXEL_COUNT))))

count = 0

for i in range(len(predictions)):
	if np.argmax(predictions[i]) != np.argmax(labels[i]):
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
	if np.argmax(predictions[i]) != np.argmax(labels[i]):
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

plt.savefig("full-incorrect-predictions.png")
