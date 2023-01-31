import cv2
import numpy as np
import dlib
from eye import Eye
import math

GREEN = (0,255,0)
THICKNESS_2 = 2
PREDICTOR_FILENAME = "shape_predictor_68_face_landmarks.dat"
CLOSED_EYE_RATIO = 4.5
OPEN_EYE_RATIO = 3.5

def real_time():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_FILENAME)
    cap = cv2.VideoCapture(0)
    is_left_eye_closed = False
    is_right_eye_closed = False
    are_eyes_closed = False
    num_of_frames_after_blink = 0

    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        num_of_frames_after_blink += 1

        faces = detector(gray)
        for face in faces:
            # add_face_rectangle(frame, face)
            landmarks = predictor(gray, face)
            left_eye = get_left_eye(landmarks)
            right_eye = get_right_eye(landmarks)
            draw_lines_for_eye(frame, left_eye)
            draw_lines_for_eye(frame, right_eye)
            left_eye_ratio = get_eye_distance_ration(left_eye)
            right_eye_ratio = get_eye_distance_ration(right_eye)
            print(left_eye_ratio, right_eye_ratio)
            # is_left_eye_closed = determine_if_eye_is_closed(is_left_eye_closed, left_eye_ratio)
            # is_right_eye_closed = determine_if_eye_is_closed(is_right_eye_closed, right_eye_ratio)

            is_left_eye_closed, is_right_eye_closed, num_of_frames_after_blink = determine_if_blinked(is_left_eye_closed, left_eye_ratio, is_right_eye_closed, right_eye_ratio, num_of_frames_after_blink)
            
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

def calculate_distance(point1, point2):
    # d = √[(x2 – x1)^2 + (y2 – y1)^2]
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_eye_distance_ration(eye):
    distance_vert = calculate_distance(eye.top_point, eye.bottom_point)
    distance_hor = calculate_distance(eye.left_point, eye.right_point)
    ratio = distance_hor/distance_vert

    return ratio

def determine_if_eye_is_closed(is_eye_closed, ratio, eye_side):
    if ratio > CLOSED_EYE_RATIO:
        # if (is_eye_closed != True):
        #     print("\n\n\n\nBLINKED " + eye_side + "\n\n\n\n")
        is_eye_closed = True
    elif ratio < OPEN_EYE_RATIO:
        is_eye_closed = False

    return is_eye_closed

def determine_if_blinked(is_left_eye_closed, left_eye_ratio, is_right_eye_closed, right_eye_ratio, num_of_frames_after_blink):
    has_blinked = False
    was_blink_long_ago = num_of_frames_after_blink > 4

    if left_eye_ratio > CLOSED_EYE_RATIO:
        if (was_blink_long_ago and is_left_eye_closed != True):
            has_blinked = True
            print("\n\n\n\nBLINKED\n\n\n\n")
            num_of_frames_after_blink = 0
        is_left_eye_closed = True
    elif left_eye_ratio < OPEN_EYE_RATIO:
        is_left_eye_closed = False

    if right_eye_ratio > CLOSED_EYE_RATIO:
        if (was_blink_long_ago and is_right_eye_closed != True and (not has_blinked)):
            has_blinked = True
            print("\n\n\n\nBLINKED\n\n\n\n")
            num_of_frames_after_blink = 0
        is_right_eye_closed = True
    elif right_eye_ratio < OPEN_EYE_RATIO:
        is_right_eye_closed = False


    return is_left_eye_closed, is_right_eye_closed, num_of_frames_after_blink

if __name__ == "__main__":
    real_time()