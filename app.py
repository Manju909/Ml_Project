# importing core pages
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os
from deepface import DeepFace

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotions = []
def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img,1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Detect face
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    #Draw Rectangle
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img , faces

def detect_emotions(our_image):
    result , faces_in_image = detect_faces(our_image)
    # Analyze the emotions of the faces using the DeepFace library
    obj = DeepFace.analyze(result, actions=['emotion'], enforce_detection=False)
    emotions = [d['dominant_emotion'] for d in obj]
    for i, emotion in enumerate(emotions):
        txt = emotion
        print("\nEmotion of ",i+1," person - ",txt)
        cv2.putText(result,txt, (50, 50*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return result


def main():
    """"Face Detection app"""
    
    st.title("Facial Emotion Detection App")
    st.text("Build with Streamlit and OpenCV")

    activities = ["Detection","about"]
    choice = st.sidebar.selectbox("Select Activities",activities)

    if choice == "Detection":

        st.subheader("Facial Detection")
        image_file = st.file_uploader("Upload Image",type = ['jpg','png','jpeg'])

        if image_file is not None :
            our_image = Image.open(image_file)
            st.text("Uploaded Image")
            st.image(our_image)

        task = ["Faces", "Emotions"]
        feature_choice = st.sidebar.selectbox("Find Features",task)
        if st.button("Process"):
            if feature_choice == 'Faces':
                result_img, result_faces = detect_faces(our_image)
                st.image(result_img)
                st.success("Found {} faces".format(len(result_faces)))
            
            elif feature_choice == 'Emotions':
                result_img2 = detect_emotions(our_image)
                st.image(result_img2)
                


    elif choice == "about":
        st.subheader("About")

if __name__ == '__main__':
    main()