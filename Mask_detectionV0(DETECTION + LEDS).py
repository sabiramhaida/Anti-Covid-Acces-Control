import cv2 # import opencv
import tensorflow.keras as keras
import numpy as np
import RPi.GPIO as GPIO
import time



np.set_printoptions(suppress=True)

#utilisation du webcam
webcam = cv2.VideoCapture(0)

#le load du model deja generé de detection de masque (classification "mask Yes/Mask No")) 
model = keras.models.load_model("MDetection.model")


data_for_model = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


#focnction qui affiche un rectangle sur l'image avec une couleur précise ( rouge ou vert dans notre cas )
# cette foction pour tester si notre model detecte le visage et si le visage detecté porte un masque ou non

def Draw_rectangle(det_face, image, color: tuple):
    for (x, y, width, height) in det_face:
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            color,
            thickness=2
        )

np.set_printoptions(suppress=True)


data = np.ndarray(shape=(1, 168, 224, 3), dtype=np.float32)

# on utilise le model par defaut de open cv qui detecte les visages!
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


#load du labels à partir d'un fichier (labels.txt)  ce fichier contient les differentes classes de notre modèle de classification
# dans notre cas nous avons deux classes ( mask Yes /NO)
def load_labels(path):
    f = open(path, 'r')
    lines = f.readlines()
    labels = []
    for line in lines:
        labels.append(line.split(' ')[1].strip('\n'))
    return labels


label_path = 'labels.txt'
labels = load_labels(label_path)

print(labels)

#cette fontion pour  redimensionner l'image 
def image_resize(image, height, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    r = height / float(h)
    dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

# focntion a pour but de ne prendre que la partie de l'image concérnée par la détection
 
def cropTo(img):
    size = 224
    height, width = img.shape[:2]

    sideCrop = (width - 224) // 2
    return img[:,sideCrop:(width - sideCrop)]


while True:

    # prendre les frames!
    ret, img = webcam.read()
    
    if ret:
    	#redimensionner et recadrer chaque frame 
        img = image_resize(img, height=224)
        img = cropTo(img)

	#detection de visage!
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_f = face_cascade.detectMultiScale(image=grayscale_image, scaleFactor=1.3, minNeighbors=4)

        #flips image
        img = cv2.flip(img, 1)

        #normalisation de l'image et la rendre en format"array" pour l'utiliser avec Keras 
        normalized_img = (img.astype(np.float32) / 127.0) - 1
        data_for_model[0] = normalized_img
 
        #executer l'interface et faire la prediction avec notre model !
        prediction = model.predict(data_for_model)
        
	#on prend les predictions pour chaque classes (Avec masque/ sans Masque)
        Avec_Mask, SANS_Mask= prediction[0]

	#Si .......... 
        if(Avec_Mask > SANS_Mask and len(detected_f)>0):
            #on affiche les probabilités ( AVEC masque/ SANS masque) 
            for i in range(0, len(prediction[0])):
                print('{}: {}'.format(labels[i], prediction[0][i]))
            #dessiner le rectangle( pour voir le resultat clairement !)) 
            Draw_rectangle(detected_f, img, (0, 255, 0))
           
        elif(Avec_Mask < SANS_Mask and len(detected_f)>0):
            for i in range(0, len(prediction[0])):
                print('{}: {}'.format(labels[i], prediction[0][i]))
            Draw_rectangle(detected_f, img, (0, 0, 255))
           

        cv2.imshow('webcam', img)
        if cv2.waitKey(1) == 27:
            break

cv2.destroyAllWindows()

