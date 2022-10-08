import torch
import tensorflow 
import cv2
import numpy as np
from PIL import Image
import torchvision
import colores
from camara2 import frames
from colores import coloreando


model = torch.hub.load('ultralytics/yolov5', 'custom', path='bestM.pt', force_reload=True)

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
        
    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret,frame =self.video.read()
        #frame1 = frame.copy()        
        detect = model(frame)    
        
        n=0
        try:
            if n>0:
                tensor = detect.xyxy[0]
                arr = tensor.cpu().detach().numpy()
                conf = arr[0][4]
            #print(conf)
                copia = frame
                frames()
                resistencia=coloreando('rotacion.jpg')
                print(resistencia)
                fuente = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, resistencia,(50, 50),fuente, 1,(0, 0,0),2, cv2.LINE_4)
            else:
                n=n+1
                cv2.imshow('Frame detectado',np.squeeze(detect.render()))
                #print(n)
        except Exception:
            pass
        fuente = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, resistencia,(50, 50),fuente, 1,(0, 0,0),2, cv2.LINE_4)

        

        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()
    
    