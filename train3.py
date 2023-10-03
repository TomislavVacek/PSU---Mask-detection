from  keras.api._v2.keras.preprocessing.image import ImageDataGenerator
from keras.api._v2.keras.layers import Conv2D, Flatten, Dropout, Dense,AveragePooling2D
from  keras.api._v2.keras.models import Sequential
from  keras.api._v2.keras.optimizers import Adam
from  keras.api._v2.keras.applications.mobilenet_v2 import preprocess_input
from  keras.api._v2.keras.preprocessing.image import img_to_array
from  keras.api._v2.keras.preprocessing.image import load_img
from  keras.api._v2.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4  ##kolike promjene smije radit, brzina,stonje manje to ce bit duze i preciznije treniranje
EPOCHS = 10
BS = 32  ##batch size- model.fit

DIRECTORY = r"Datasets/Train"	##ucitavanje baze podataka u program
CATEGORIES = ["withMask", "withoutMask"]


print("[INFO] loading images...")

data = []  ##inicijaliziramo listu
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(category)
#konventiranje
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#numpy array-polje
data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
##ako zakrenem glavu pa da moze svejedno prepoznat
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
##################################
##treniranje modela
##mreza,najbitnije... pogledaj sve....conv2D itd...
model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=(224,224,3))) ##2D convolution layer (prostorna konvulacija preko slike)
model.add(AveragePooling2D(pool_size=(7, 7))) 	##Prosječna operacija skupljanja prostornih podataka
model.add(Flatten(name="flatten"))			  	##Vrati kopiju polja sažetog u jednu dimenziju
model.add(Dense(1024, activation="relu"))	 	##Dense implements the operation
model.add(Dropout(0.5))	##The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting
model.add(Dense(2, activation="softmax"))


print("compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)  						##optimizer, 
model.compile(loss="binary_crossentropy", optimizer=opt,   				##nadi sta smo u labosima koristili od optimizatora
	metrics=["accuracy"])

################################
print("trainin model...")
H = model.fit(  												 ##bitno model fit: treniranje modela
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

print("training finished...")
predIdxs = model.predict(testX, batch_size=BS)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

print("saving model...")
model.save("maskModel.model", save_format="h5")
################
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")