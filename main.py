import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.datasets import mnist
#from tensorflow.python.keras import layers, models
#from keras.datasets import mnist

# erstelle datenset mit training images und training labels sowie testing images und testing labels, diese werden vom Tensorflow bereitgestellt
#training und testing img sind einfache arrays of pixels

(training_img, training_label), (testing_img, testing_label) = keras.datasets.cifar10.load_data()

# runterskalieren der images sodass die values der pixel im image nicht von 0 - 255 sondern nur von 0-1 gehen

training_img, testing_img = training_img / 255, testing_img / 255

# Da die Label aus dem keras datenset leider nur nummern sind müssen wir also selbst eine Stringzuordnung zur Labelnummer machen
#dafür erstellen wir eine Liste, leider sind die nummern aus den Datensatz fix, d.h. nummer 1 ist immer plane nummer 2 ist immer car, usw..

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# erstellen eines grids welches 16 Bilder itriert

for i in range(16):
    #erstellt ein 4x4 grid, mit jeder iteration wird ien element im grid befüllt
    plt.subplot(4,4,i+1)
    # wir wollen kein koordinatensystem nutzen daher setzen wir x und y ticks auf nichts
    plt.yticks([])
    plt.xticks([])
    #anzeigen des images + der sogenannten colormap
    plt.imshow(training_img[i], cmap=plt.cm.binary)
    #dem bild ein Label hinzufügen, wir nutzen unsere obere class_names liste dafür
    plt.xlabel(class_names[training_label[i][0]])

plt.show()




#da ein vergleich mit allen daten im bereitgestellten datensatz lange dauerd reduzieren wir den Datensatz auf 20000 bilder
#um mehr datensätze umso höher ist natürlich die genauigkeit, für unsere bedürfnisse reicht einreduzierter datensatz
training_img = training_img[:10000]
training_label = training_label[:10000]


# das gleiche mal mit den testing images und labels auf 4000

testing_img = testing_img[:10000]
testing_label = testing_label[:10000]




# Nun erstellen wir das "neuronale Netzwerk"

model = keras.models.Sequential()

# nun definieren wir mehrere input layer die layer werden nacheinander durchlaufen
#hierbei handelt es sich um ein convolution over images Layer, TODO mehr über die verschiednen Layer lernen
#input shape gibt (pixel, pixel, Farbkanäle) Farbkanäle 3 für RGB

#Ein Convolution Layer klassifiert elemente eines Bildes (z.b. Mensch = hat beine, hat arme ... )
model.add(layers.Conv2D(32,(3,3),activation='relu', input_shape=(32,32,3)))

#bei jeden Conv2D Layer ist auch ein MaxPooling2D Layer einzufügen da dieser die zurückgegebene Informationen versimpelt
# auf die kerninformation reduziert
model.add(layers.MaxPooling2D((2,2)))

# nochmal ein Conv2D Layer, diemal mit einer filtergröße von 64
model.add(layers.Conv2D(64,(3,3),activation='relu', ))
# immer nach einem Conv2D Maxpooling..
model.add(layers.MaxPooling2D((2,2)))
# erneut Conv2D
model.add(layers.Conv2D(64,(3,3),activation='relu', ))

#da jetzt die Ergebnisse bereitstehen wollen wir diese in einer dimension darstellen, das result der layer ist als grid gesichert
# wir möchten dennoch das die ergebnisse nicht in einem z.b. 10x10 array sondern in einem eindimensionalen Array dargestellt werden (liniear 100 einträge)

model.add(layers.Flatten())

#Vermutung: da wir 64 Filter einsetzen und als Ergebniss an dieser Stelle erhalten wird auch hier 64 Units angegeben
model.add(layers.Dense(64, activation="relu"))

#Da wir 10 verschiedene Klassifikationon haben (siehe class_names) wird hier 10 angegeben
model.add(layers.Dense(10, activation="softmax"))


# nun müssen wird nur noch das trainierte model generieren

model.compile(optimizer="adam",loss='sparse_categorical_crossentropy', metrics=["accuracy"])

#hier erklären wir dem model das er die selben Datensätze 10 mal itriert (also jedes bild 10 mal trainiert)
# Es wird angegeben welche daten für das training verwendet werden sollen
# Es wird unter validierung testdaten mit der richtigen zuordnung benötigt
model.fit(training_img, training_label, epochs=100, validation_data=(testing_img, testing_label))





#