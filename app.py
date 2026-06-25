import streamlit as st
import cv2
import mediapipe as mp
from scipy.spatial import distance

st.title("Driver Drowsiness Detection System")

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True
)

LEFT_EYE=[33,160,158,133,153,144]
RIGHT_EYE=[362,385,387,263,373,380]


def eye_aspect_ratio(points, landmarks,w,h):

    pts=[]

    for p in points:
        x=int(landmarks[p].x*w)
        y=int(landmarks[p].y*h)
        pts.append((x,y))

    A=distance.euclidean(pts[1],pts[5])
    B=distance.euclidean(pts[2],pts[4])
    C=distance.euclidean(pts[0],pts[3])

    return (A+B)/(2*C)


def mouth_aspect_ratio(landmarks,w,h):

    top=(
        int(landmarks[13].x*w),
        int(landmarks[13].y*h)
    )

    bottom=(
        int(landmarks[14].x*w),
        int(landmarks[14].y*h)
    )

    left=(
        int(landmarks[78].x*w),
        int(landmarks[78].y*h)
    )

    right=(
        int(landmarks[308].x*w),
        int(landmarks[308].y*h)
    )

    vertical=distance.euclidean(
        top,bottom
    )

    horizontal=distance.euclidean(
        left,right
    )

    return vertical/horizontal


run = st.checkbox("Start Webcam")

frame_window = st.image([])

score=0

cap=cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

while run:

    ret,frame=cap.read()

    if not ret:
        st.write("Camera Error")
        break

    h,w,_=frame.shape

    rgb=cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results=face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            landmarks=face_landmarks.landmark

            leftEAR=eye_aspect_ratio(
                LEFT_EYE,
                landmarks,w,h
            )

            rightEAR=eye_aspect_ratio(
                RIGHT_EYE,
                landmarks,w,h
            )

            ear=(leftEAR+rightEAR)/2

            mar=mouth_aspect_ratio(
                landmarks,w,h
            )

            if ear < 0.25 or mar > 0.35:
                score +=1
            else:
                score=max(0,score-1)


            if score >=5:

                cv2.putText(
                    frame,
                    "DROWSY",
                    (500,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2
                )


            cv2.putText(
                frame,
                f"EAR:{ear:.2f}",
                (10,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"MAR:{mar:.2f}",
                (10,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"Score:{score}",
                (10,120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )


    frame_window.image(
        cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )
    )

cap.release()