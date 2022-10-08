from flask import Flask,render_template,Response
from camara import Video
import numpy as np
import cv2

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('antonio_index.html')

def gen(camara):
    while(1):
        frame=camara.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')
        if cv2.waitKey(5) & 0xFF == ord('d'):
            break

@app.route('/video')

def video():
    return Response(gen(Video()),  
    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)