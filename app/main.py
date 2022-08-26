# import requirements needed
from flask import Flask, render_template, Response, render_template_string, request, make_response, redirect, url_for
import numpy as np
import datetime
import json
import os

from flask_utils import get_base_url
import cv2
import random
import torch
import copy

#vid = cv2.VideoCapture(0)
temp = (240, 320, 3)
img = np.zeros(temp, np.uint8)
size = 20
color = (0, 0, 255)
bar = temp[0] - 60
seg = bar/6

background_image_flag = False


# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12345
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# set up the routes and logic for the webserver
@app.route(f'{base_url}')
def home():
    return render_template('index.html')


@app.route(f'{base_url}/test.html')
def canvas():
    return render_template('test.html', width=320, height=240)

def send_file_data(data, mimetype='image/jpeg', filename='output.jpg'):
    # https://stackoverflow.com/questions/11017466/flask-to-return-image-stored-in-database/11017839

    response = make_response(data)
    response.headers.set('Content-Type', mimetype)
    response.headers.set('Content-Disposition', 'attachment', filename=filename)

    return response

def changeSize(i):
                global size
                if((size > 10 and i < 0) or (size < 50 and i > 0)):
                    size = size + i
def drawCircle(x, y):
    global color
    cv2.circle(img, (x,y), size, color, -1)

def changeMode(x):
    mode = x
                
@app.route(f'{base_url}/upload', methods=['GET', 'POST'])
def upload():
    
    global color
    
    if request.method == 'POST':
         
        # Get the file from the website post request
        fs = request.files.get('snap')
        if fs:
            
            
            # Turns byte stream into image
            frame = cv2.imdecode(np.frombuffer(fs.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            
            # Get the model predictions
            results = model(frame)
            
            #######DRAWING CODE
            output = results.pandas().xyxy[0]
            if(output.shape[0] == 1):
                cv2.rectangle(frame, (int(output.xmin), int(output.ymin)),(int(output.xmax), int(output.ymax)), color, 4)
                cv2.putText(frame, str(output.name[0]), (int(output.xmin), int(output.ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

                x = int((output.xmin + output.xmax)/2)
                y = int((output.ymin + output.ymax)/2)
                if(str(output.name[0]) == 'thumbsUp'):

                    changeSize(1)
                    cv2.putText(frame, str(size), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                elif(str(output.name[0]) == 'thumbsDown'):
                    changeSize(-1)
                    cv2.putText(frame, str(size), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                elif(str(output.name[0]) == 'palm'):
                    color = (0,0,0)
                    cv2.putText(frame, 'Erase', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
                elif(str(output.name[0]) == 'fist'):
                    if(0 < x and x < 100):
                        z = y-40
                        if(z < 0 or z > bar):
                            color = (255,255,255)
                        else:
                            per = z%seg * 1.0/seg
                            if(int(z/seg) == 0):
                                color = (0, int(per * 255), 255)
                            if(int(z/seg) == 1):
                                color = (0, 255, int((1-per) * 255))
                            elif(int(z/seg) == 2):
                                color = (int(per * 255), 255, 0)
                            elif(int(z/seg) == 3):
                                color = (255, int((1-per) * 255), 0)
                            elif(int(z/seg) == 4):
                                color = (255, 0, int(per * 255))
                            elif(int(z/seg) == 5):
                                color = (int((1-per) * 255), 0, 255)
                    cv2.putText(frame, str(color), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                elif(str(output.name[0]) == 'point'):
                    drawCircle(int(output.xmin), int(output.ymin))
              
            #flag whether there is an uploaded background image or not
            
            
            
             
            # Overlays drawing (img) onto webcam (frame)
            
            # result is weighted sum of frame + img ONLY if image_st_fin is not set yet
            if not background_image_flag:
                result = cv2.addWeighted(frame, 1, img, 1, 0)
            else: # otherwise, result is weighted sum of image_st_fin + image
                result = cv2.addWeighted(image_st_fin, 1, img, 1, 0)
            
            
            
            #######END DRAWING CODE
            
            # Encode the img to a jpeg and turn back into byte stream
            ret, buf = cv2.imencode('.jpg', result)

            # Call send_file_data function to send the modified image back to the website display
            return send_file_data(buf.tobytes())
        else:
            return 'You forgot Snap!'

    return 'Hello World!'



image_st_fin = None
@app.route(f'{base_url}/test.html', methods = ['POST'])
def upload_file():
    print("hello")
    if request.method == 'POST':
        
        # Get the file from post request
        f = request.files['filename']
        print(f)
        
        # If a file exists
        if f.filename:
            
            # Save the file to some folder under f.filename
            f.save(f.filename)

            # Open the file (image) from path with cv2
            
            image_st = cv2.imread(f.filename)

            # Resize the image to be (240, 320) so that it is the same size as drawing
            
            # Save it to some global variable to be used later
            global image_st_fin
            image_st_fin = cv2.resize(image_st,(320, 240))
            
            global background_image_flag
            background_image_flag = True

        
      
        return redirect(url_for('canvas'))

##WEBCAM CODE
camera = cv2.VideoCapture(0)
'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''
def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route(f'{base_url}/test.html/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc15.ai-camp.dev/'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)
    
    

