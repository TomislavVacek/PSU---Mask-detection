from asyncio.windows_events import NULL
from keras.api._v2.keras.applications.mobilenet_v2 import preprocess_input
from keras.api._v2.keras.preprocessing.image import img_to_array
from keras.api._v2.keras.models import load_model
import numpy as np
import cv2


##pronalazenje lica
def detectFaces(frame, faceModel):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),             ##blob from image
	(104.0, 177.0, 123.0))
    (h, w) = frame.shape[:2] 
    faceModel.setInput(blob)               ##predavanje bloba, frame koji je malo preoblikovan
    detections = faceModel.forward()       ##objekt koji sve sadrzi,koordinate lica...
    foundFaces = []                        ## otvaranje arraya za upisivanje koordinata od lica
    for i in range(0, detections.shape[2]):##prolazi korz svako lice koje je naslo u frameu
        if detections[0,0,i,2] > 0.3:      ##ako je vece oda 30% onda ulazim u if
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) ##spremam koord od detekcije
            (startX, startY, endX, endY) = box.astype("int")        ##koordinate ##za okvira ##sprema u obliku int
            (startX, startY) = (max(0, startX), max(0, startY))     ##224x224
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            foundFaces.append((startX, startY, endX, endY))          ##dodavanje u array-append
    return foundFaces




def detectMask(frame, foundFaces, maskModel):
    predictionList = []  ##inicijalizacija prediction liste
    faces = []          ##inicijalizacija lica
    for (startX,startY,endX,endY) in foundFaces: 
        
        face = cv2.resize(frame[startY:endY,startX:endX], (224, 224)) ##resizea jer mask model prima 224x224 slike x3
        face = img_to_array(face)                                     ##matrica pretvorimo u array
        face = preprocess_input(face)                                 ##predobrada lica
        faces.append(face)                                            ##obradene podatek sprema u face

    faces = np.array(faces, dtype="float32")                  ##matricu ce prebacit u array
    predictionList = maskModel.predict(faces, batch_size=32) ##predajes array modelu i model vraca array predikcije
    return predictionList                                    ##sa ili bez maske(sa maskom % , bez maske %),(vraca sa maskom,bez maske...)

def showResultsOnVideo():

    faceModel = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel") ##loadanje modela,ucitavanje...
    maskModel = load_model("maskModel.model")                                                   ##ucitavanje mask modela

    cap = cv2.VideoCapture(0) ##dobivanje inputa sa kamere, 0-prva pronadena kamera
    color=(255,0,0)
    stroke = 2                  #debljina linije

    while True:

        ret, frame = cap.read()                     ##PROCITA STA JE KAMERA UCITALA PA SPREMI U FRAME, FRAME PROSLIEJDNIM U DETECT MASK I DETECET FACE
        foundFaces = detectFaces(frame,faceModel)
             
        if foundFaces!=[]:                                           ##ako je naso bar 1 lice onda ulazi u if
            
            predictionList = detectMask(frame,foundFaces,maskModel)
            i = 0
            for (startX,startY,endX,endY) in foundFaces:

                (withoutMask,withMask)=predictionList[i] 
                maskConf = int(withMask*100) if withMask > withoutMask else int(withoutMask*100)  ##ispisivanje teksta u kutu
                maskConf = str(maskConf)
                text = "With mask" if withMask > withoutMask else "Without mask"
                text = text + " " + maskConf + "%"                                  ##pise u kutu ,postotak i pise jel imas ili nemas
                cv2.putText(frame, text, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame,(startX,startY),(endX,endY),color,stroke)
                i = i+1

        cv2.imshow('frame',frame)
        if(cv2.waitKey(1)==ord('q')):
            break


showResultsOnVideo()