from imutils.face_utils import FaceAligner
from imutils.video import count_frames
import imutils
import numpy as np
import dlib
import time
import cv2

def load_caffe_models():
    path = 'D:\\Webcam_Face_Detection\\'
    # Age predict NN model
    age_net = cv2.dnn.readNetFromCaffe(path+'deploy_age.prototxt', 'age_net.caffemodel')
    # Gender predict NN model
    gender_net = cv2.dnn.readNetFromCaffe(path+'deploy_gender.prototxt', 'gender_net.caffemodel')

    return(age_net, gender_net)

# Load Dlib Face Detection Model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# DNN
age_net, gender_net = load_caffe_models() # 預測模型讀取(年齡、性別)

video_capture = cv2.VideoCapture(0) # open default webcam
video_capture.set(3, 480) #set width of the frame
video_capture.set(4, 640) #set height of the frame 

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# range of age prediction 
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

gender_list = ['Male', 'Female']

while True :
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1) # 影片鏡面呈現
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
  
    for face in faces :

        x = face.left()
        y = face.top()
        w = face.right() - face.left()
        h = face.bottom() - face.top()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2) # (0, 255, 0), 2)
        
# Get Face 
    face_img = frame[y:y+h, h:h+w].copy()
    blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

# Predict Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    print("Gender : " + gender)

#Predict Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]
    print("Age Range: " + age)

    overlay_text = "%s %s" % (gender, age)
    cv2.putText(frame, overlay_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) , 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)  
#0xFF is a hexadecimal constant which is 11111111 in binary.
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
video_capture.release()
cv2.destroyAllWindows()
