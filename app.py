import streamlit as st
import pandas as pd
import numpy as np


# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import time 
#For arduino connection
import serial
from PIL import Image
from customMsg import customMsg, customMsg2


#For arduino connection
arduino = serial.Serial('/dev/cu.usbserial-1410', 9600)
                             
lowConfidence = 0.75
def detectAndPredictMask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > lowConfidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
    if len(faces) > 0:
       faces = np.array(faces, dtype="float32")
       preds = maskNet.predict(faces, batch_size=32)        
    return (locs, preds)


def main():
    #setting custom Page Title and Icon with changed layout and sidebar state
    icon = Image.open("static/icon.png")
    st.set_page_config(page_title='Face Mask Entrance Control System', layout='wide', initial_sidebar_state='expanded', page_icon=icon)
    footer="""<style>

    #MainMenu{visibility:hidden;}
    footer{visibility:hidden;}

    .footer {
    position:fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    z-index:2;

    }
    </style>
 <div class="footer">
                    <p class="footer-text m-0">
                      © 2022 | Developed by 
                      <a href="https://github.com/sonamehdi19" target="_blank"
                        >sonamehdi19</a
                      >
                    </p>
  </div>

    """
    st.markdown(footer,unsafe_allow_html=True)

    ban = Image.open("static/facemask.jpg")
    lg= Image.open("static/logo.png")
    st.sidebar.image(lg, use_column_width="always")

    
    pg = st.sidebar.selectbox("", ["Homepage", "Application"])
    if pg == "Homepage":
        st.markdown('<h1 align="center"> Face Mask Entrance Control System</h1>', unsafe_allow_html=True)
        st.image(ban, use_column_width="always")
        st.header("Information about project")
        st.markdown("""This application is developed in order to automate the face mask checking at the entrances of workplaces in order to minimize the human intervention and also save time. 
            It is the final project for the Smart Sensors and Systems.
            If the entrance is granted based on face mask, people will be directed to hand sanitizer.""", unsafe_allow_html=True)
    else:
        st.markdown('<h2 align="center">Welcome!</h2>', unsafe_allow_html=True)

        prototxtPath = r"/Users/sonamehdizade/Desktop/Face-Mask-Detection-Based-Door-Lock-System/deploy.prototxt"
        weightsPath = r"/Users/sonamehdizade/Desktop/Face-Mask-Detection-Based-Door-Lock-System/res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        maskNet = load_model("/Users/sonamehdizade/Desktop/Face-Mask-Detection-Based-Door-Lock-System/mask_detector.model")

        FRAME_WINDOW=st.image([])

        vs = VideoStream(src=0).start()
        while True:
            status=st.empty()
            frame = vs.read()
            frame = imutils.resize(frame, width=900)
            (locs, preds) = detectAndPredictMask(frame, faceNet, maskNet)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                if label =="Mask":
                    msg = 'ACCESS GRANTED'
                    print(msg);
                    #status.success(msg);
                    customMsg2(msg, 0.1, 'success')
                    #For arduino connection
                    arduino.write(b'H')
                else:
                    msg = 'ACCESS DENIED';
                    print(msg)
                    customMsg2(msg, 0.1, 'warning')
                    #status.warning(msg)
                    #For arduino connection
                    arduino.write(b'L')

                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
                status.empty()

            cv2.imshow("Press q to quit", frame)  
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # wait for ESC key to exit

        vs.stop()
    cv2.destroyAllWindows()
    print('Exited from process')

if __name__ == "__main__":
    main()
