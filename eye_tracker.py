import cv2
import numpy as np
import dlib
from eye import Eye

GREEN = (0,255,0)
THICKNESS_2 = 2
PREDICTOR_FILENAME = "shape_predictor_68_face_landmarks.dat"



def real_time():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_FILENAME)
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            # add_face_rectangle(frame, face)
            landmarks = predictor(gray, face)
            left_eye = get_left_eye(landmarks)
            right_eye = get_right_eye(landmarks)
            draw_lines_for_eye(frame, left_eye)
            draw_lines_for_eye(frame, right_eye)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cap.destroAllWindows()

def add_face_rectangle(frame, face):
    x1,y1 = face.left(), face.top()
    x2, y2 = face.right(), face.bottom()
    cv2.rectangle(frame, (x1,y1), (x2,y2), GREEN, THICKNESS_2)

def get_landmark_point(landmarks, point_number):
    return (landmarks.part(point_number).x, landmarks.part(point_number).y)

def get_point_between_two_points(point1, point2):
    x = int((point1[0] + point2[0]) / 2) # int is for rounding
    y = int((point1[1] + point2[1]) / 2)

    return (x,y)

def get_left_eye(landmarks):
    left_point = get_landmark_point(landmarks, 36)
    right_point = get_landmark_point(landmarks, 39)

    top_left_point = get_landmark_point(landmarks, 37)
    top_right_point = get_landmark_point(landmarks, 38)
    top_point = get_point_between_two_points(top_left_point, top_right_point)

    bottom_left_point = get_landmark_point(landmarks, 41)
    bottom_right_point = get_landmark_point(landmarks, 40)
    bottom_point = get_point_between_two_points(bottom_left_point, bottom_right_point)

    eye = Eye(left_point, right_point, top_point, bottom_point)
    return eye

def get_right_eye(landmarks):
    left_point = get_landmark_point(landmarks, 42)
    right_point = get_landmark_point(landmarks, 45)

    top_left_point = get_landmark_point(landmarks, 43)
    top_right_point = get_landmark_point(landmarks, 44)
    top_point = get_point_between_two_points(top_left_point, top_right_point)

    bottom_left_point = get_landmark_point(landmarks, 47)
    bottom_right_point = get_landmark_point(landmarks, 46)
    bottom_point = get_point_between_two_points(bottom_left_point, bottom_right_point)

    eye = Eye(left_point, right_point, top_point, bottom_point)
    return eye

def draw_lines_for_eye(frame, eye):
    horizontal_line = cv2.line(frame, eye.left_point, eye.right_point, GREEN, THICKNESS_2)
    vertical_line = cv2.line(frame, eye.top_point, eye.bottom_point, GREEN, THICKNESS_2)

if __name__ == "__main__":
    real_time()