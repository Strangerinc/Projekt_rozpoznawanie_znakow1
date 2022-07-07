import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlt
import cv2
import os
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from platform import python_version
########################################################################
# Jeśli import wtyczek zakończył sie błedem z informacja o braku zainstalowanej wtyczki
# przekopiuj poniższa linię bez # do terminala i uruchom ją
# pip install numpy scipy matplotlib ipython jupyter pandas sympy nose scikit-learn opencv-python tensorflow scikit-learn opencv-python tensorflow
########################################################################

##################### Sprawdzamy wersję zainstalowanych wtyczek ########
print("############### Mamy nastepujace wersje wtyczek #####")
print("Python version: ",python_version())
print("Numpy version: ",np. __version__)
print("Patplotlib version: ",mlt. __version__)
print("Opencv version: ",cv2. __version__)
print("Pandas version: ",pd. __version__)
print("Tensorflow version: ",tf. __version__)
print("#############################################################")
################# Parametry naszego projektu  #####################

path = "c:/Users/Stranger_inc/Downloads/!!!PROJEKT!!!/myData"  # folder with all the class folders
labelFile = 'c:/Users/Stranger_inc/Downloads/!!!PROJEKT!!!/labels.csv'  # file with all names of classes
batch_size_val = 50  # Jak dużo zdjeć zabrać ze zbioru treningowego na epokę
epochs_val = 30   # ustalamy ilość epok minimum 15 coć po 15 epoce nadal wykres rośnie
imageDimesions = (32, 32, 3)  # 1024 pixele
testRatio = 0.2  # 20% ze zbioru treningowego na kazde 100 da nam 20 jako zbiór testowy
validationRatio = 0.2  # 20 % z pozostałych 80 zbioru treningowego czyli 16 zbiór validacyjny
###################################################


############################### Sekcja odpowiedzialna za wczytanie obeazków #############
count = 0  # 0 uzyjemy jako katalogu startowego i bedziemy go zwiększać o kolejną wczytaną klasę
images = []   # lista obrazków
classNo = []  # lista wszystkich naszych klass
myList = os.listdir(path)  # Return a list containing the names of the files in the directory.
print("Ilość wykrytych klas:", len(myList))  # długść listy na podstawie zawartości katalogu
noOfClasses = len(myList)  # do ilości klas przypisuję zmienną noOfClasses
print("Importing Klas.....")
for x in range(0, len(myList)):  # pętla która ma za zadanie wczytanie kolejnych katalogów z danymi
    myPicList = os.listdir(path + "/" + str(count))  # patch podaje dostęp do katalogu MyData count podstawia  nr kolejnego katalogu
    for y in myPicList:  # pętla w pętli  wczytujaca zawartośc poszczególnych katalogów w MyData
        curImg = cv2.imread(path + "/" + str(count) + "/" + y)  # Wczytujemy kolejne obrazki z MyData
        images.append(curImg)   #
        classNo.append(count)   # zwiększamy listę klas o kolejną klasę
    print(count, end=" ")
    count += 1   # zwiększamy count o 1 co spowoduje przejście do kolejnego katalogu
print(" ")
images = np.array(images)  # budujemy macierz z oobrazkami
classNo = np.array(classNo)  # budujemy macierz z klasami

############################### Split Data - Podział naszych danych na treningowe, testowe i walidacyjne ###
Image_train, Image_test, label_train, label_test = train_test_split(images, classNo, test_size=testRatio)
Image_train, Image_validation, label_train, label_validation = train_test_split(Image_train, label_train, test_size=validationRatio)

# Image_train = ARRAY OF IMAGES TO TRAIN
# label_train = CORRESPONDING CLASS ID

############################### SPRAWDZAMY CZY LICZBA OBRAZÓW JEST ZGODNA Z LICZBĄ ETYKIET DLA KAŻDEGO ZBIORU DANYCH
print("Data Shapes")
print('All Data Sets:', len(images))
print("Train", end="");
print(Image_train.shape, label_train.shape)
print("Validation", end="");
print(Image_validation.shape, label_validation.shape)
print("Test", end="");
print(Image_test.shape, label_test.shape)
assert (Image_train.shape[0] == label_train.shape[0]), "Liczba obrazów nie jest równa liczbie etykiet w zestawie uczącym"
assert (Image_validation.shape[0] == label_validation.shape[0]), "Liczba obrazów nie jest równa liczbie etykiet w zestawie walidacyjnym"
assert (Image_test.shape[0] == label_test.shape[0]), "Liczba obrazów nie jest równa liczbie etykiet w zestawie testowym"
assert (Image_train.shape[1:] == (imageDimesions)), " Wymiary obrazów szkoleniowych są nieprawidłowe"
assert (Image_validation.shape[1:] == (imageDimesions)), " Wymiary obrazów walidacji są nieprawidłowe "
assert (Image_test.shape[1:] == (imageDimesions)), "Wymiary obrazów testowych są nieprawidłowe "

############################### WCZYTUJEMY NASZ PLIK CSV Z NASZYMI LABELKAMI #############
data = pd.read_csv(labelFile)  # we read our labels
print("data shape ", data.shape, type(data))  # Wyświetlamy jak wygląda nasz Label files ( ile zawiera klas i ile obiektów)

############################### WYŚWIETLAMY PRZYKŁADOWE OBRAZKI ZE WSZYSTKICH KLAS
num_of_samples = []
cols = 5
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = Image_train[label_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))

############################### DISPLAY A BAR CHART SHOWING NO OF SAMPLES FOR EACH CATEGORY
print(num_of_samples)  # wypisuje ile jest obrazków w danej klasie
plt.figure(figsize=(12, 4))  # ustalamy wielkośc naszego wykresu
plt.bar(range(0, num_classes), num_of_samples)   # rysować wykres dla klas z zakresu od 0 do num_of_classes biorąc pod uwagę ilośc obrazków w klasie
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()


############################### PREPROCESSING THE IMAGES

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)  # CONVERT TO GRAYSCALE
    img = equalize(img)  # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img / 255  # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img

########################### TO IRETATE AND PREPROCESS ALL IMAGES
Image_train = np.array(list(map(preprocessing, Image_train)))
Image_validation = np.array(list(map(preprocessing, Image_validation)))
Image_test = np.array(list(map(preprocessing, Image_test)))

cv2.imshow("GrayScale Images",
           Image_train[random.randint(0, len(Image_train) - 1)])  # TO CHECK IF THE TRAINING IS DONE PROPERLY

############################### ADD A DEPTH OF 1
Image_train = Image_train.reshape(Image_train.shape[0], Image_train.shape[1], Image_train.shape[2], 1)
Image_validation = Image_validation.reshape(Image_validation.shape[0], Image_validation.shape[1], Image_validation.shape[2], 1)
Image_test = Image_test.reshape(Image_test.shape[0], Image_test.shape[1], Image_test.shape[2], 1)

############################### AUGMENTATAION OF IMAGES: TO MAKEIT MORE GENERIC
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             # 0.1 = 10%     IF MORE THAN 1 E.G 10 THEN IT REFFERS TO NO. OF  PIXELS EG 10 PIXELS
                             height_shift_range=0.1,
                             zoom_range=0.2,  # 0.2 MEANS CAN GO FROM 0.8 TO 1.2
                             shear_range=0.1,  # MAGNITUDE OF SHEAR ANGLE
                             rotation_range=10)  # DEGREES
dataGen.fit(Image_train)
batches = dataGen.flow(Image_train, label_train,
                       batch_size=32)  # REQUESTING DATA GENERATOR TO GENERATE IMAGES  BATCH SIZE = NO. OF IMAGES CREAED EACH TIME ITS CALLED
X_batch, y_batch = next(batches)

# TO SHOW AGMENTED IMAGE SAMPLES
fig, axs = plt.subplots(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0], imageDimesions[1]))
    axs[i].axis('off')
plt.show()

label_train = to_categorical(label_train, noOfClasses)
label_validation = to_categorical(label_validation, noOfClasses)
label_test = to_categorical(label_test, noOfClasses)


############################### CONVOLUTION NEURAL NETWORK MODEL
def myModel():
    no_Of_Filters = 60  # ZESTAW FILTROW KTÓRE MAJĄ SZUKAĆ
    size_of_Filter = (5, 5)  # THIS IS THE KERNEL THAT MOVE AROUND THE IMAGE TO GET THE FEATURES.
    # THIS WOULD REMOVE 2 PIXELS FROM EACH BORDER WHEN USING 32 32 IMAGE
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)  # SCALE DOWN ALL FEATURE MAP TO GENERALIZE  MORE, TO REDUCE OVERFITTING
    no_Of_Nodes = 500  # NO. OF NODES IN HIDDEN LAYERS

    model = Sequential()
    # 60 FILTRÓW i kernel (5,5) 1500 parametrów do nauki
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(imageDimesions[0], imageDimesions[1], 1),
                      activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))  # DOES NOT EFFECT THE DEPTH/NO OF FILTERS
    # 30 FILTÓW i kernel (3,3) i tu już 270 parametrów do nauki
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
    # operacja spłaszczenia zmiana wymiarów obrazka do jednego wymiaru
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))  # INPUTS NODES TO DROP WITH EACH UPDATE 1 ALL 0 NONE
    model.add(Dense(noOfClasses, activation='softmax'))  # OUTPUT LAYER ( zwraca odpowiednia klasę i jej numer )
    # COMPILE MODEL
    model.compile(Adam(learning_rate = 0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


############################### TRAIN
model = myModel()
print(model.summary())
history = model.fit(dataGen.flow(Image_train, label_train, batch_size=batch_size_val), epochs=epochs_val,
                    validation_data=(Image_validation, label_validation), shuffle=1)

############################### PLOT
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(Image_test, label_test, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# STORE THE MODEL AS A PICKLE OBJECT
keras.models.save_model(model, "c:/Users/Stranger_inc/Downloads/!!!PROJEKT!!!/my_model_test2.hdf5")

cv2.waitKey(0)
