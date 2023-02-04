import cv2
import numpy as np
import dlib
from eye import Eye
import math
import os

GREEN = (0,255,0)
THICKNESS_2 = 2
PREDICTOR_FILENAME = "shape_predictor_68_face_landmarks.dat"
CLOSED_EYE_RATIO = 4.5
OPEN_EYE_RATIO = 3.5

def setup_eyes_vars():
    is_left_eye_closed = False
    is_right_eye_closed = False
    are_eyes_closed = False
    num_of_frames_after_blink = 0
    num_of_blinks = 0

    return is_left_eye_closed, is_right_eye_closed, are_eyes_closed, num_of_frames_after_blink, num_of_blinks 

def setup_gaze_vars():
    num_of_frames_after_gaze_direction_change = 0
    gaze_direction = "CENTER"
    gazes_to_left = 0
    gazes_to_right = 0

    return gaze_direction, num_of_frames_after_gaze_direction_change, gazes_to_left, gazes_to_right

def real_time(threshold_level):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_FILENAME)
    cap = cv2.VideoCapture(0)
    # is_left_eye_closed = False
    # is_right_eye_closed = False
    # are_eyes_closed = False
    # num_of_frames_after_blink = 0
    # num_of_frames_after_gaze_direction_change = 0
    # gaze_direction = "CENTER"
    # gazes_to_left = 0
    # gazes_to_right = 0
    is_left_eye_closed, is_right_eye_closed, are_eyes_closed, num_of_frames_after_blink, num_of_blinks = setup_eyes_vars()
    gaze_direction, num_of_frames_after_gaze_direction_change, gazes_to_left, gazes_to_right = setup_gaze_vars()
    # num_of_blinks = 0
    is_active = True

    while is_active:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        num_of_frames_after_blink += 1
        num_of_frames_after_gaze_direction_change += 1

        faces = detector(gray)
        for face in faces:
            # add_face_rectangle(frame, face)
            landmarks = predictor(gray, face)
            left_eye = get_left_eye(landmarks)
            right_eye = get_right_eye(landmarks)
            # draw_lines_for_eye(frame, left_eye)
            # draw_lines_for_eye(frame, right_eye)
            left_eye_ratio = get_eye_distance_ration(left_eye)
            right_eye_ratio = get_eye_distance_ration(right_eye)
            # print(left_eye_ratio, right_eye_ratio)

            is_left_eye_closed, is_right_eye_closed, num_of_frames_after_blink, has_blinked = determine_if_blinked(is_left_eye_closed, left_eye_ratio, is_right_eye_closed, right_eye_ratio, num_of_frames_after_blink)
            
            if (has_blinked):
                num_of_blinks += 1
            else:
                left_eye_region = get_only_eye_region(left_eye)
                threshold_gray_eye_frame = get_eye_gaussed_thresholded_frame(left_eye_region, frame, threshold_level)
                cv2.imshow("Eye frame", threshold_gray_eye_frame)

                left_part, center_part, right_part = split_eye_frame_into_three_parts(threshold_gray_eye_frame)
                show_eye_parts_frames(left_part, center_part, right_part)
  
                white_on_left_part, white_on_center_part, white_on_right_part = count_white_surface(left_part, center_part, right_part)

                # print(white_on_left_part, white_on_center_part, white_on_right_part)
                gaze_direction, increase_left, increase_right, num_of_frames_after_gaze_direction_change = determine_gaze_direction(gaze_direction, white_on_left_part, white_on_center_part, white_on_right_part, num_of_frames_after_gaze_direction_change, num_of_frames_after_blink)
                gazes_to_left += increase_left
                gazes_to_right += increase_right

            print("BLINKS: " + str(num_of_blinks) + "   " + "LEFT: " + str(gazes_to_left) + "   " + "RIGHT: " + str(gazes_to_right))
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            is_active = False

    cap.release()
    cap.destroAllWindows()

def get_only_eye_region(left_eye:Eye):
    eye_region = np.array([(left_eye.left_point[0], left_eye.left_point[1]),
                                (left_eye.top_left_point[0], left_eye.top_left_point[1]),
                                (left_eye.top_right_point[0], left_eye.top_right_point[1]),
                                (left_eye.right_point[0], left_eye.right_point[1]),
                                (left_eye.bottom_right_point[0], left_eye.bottom_right_point[1]),
                                (left_eye.bottom_left_point[0], left_eye.bottom_left_point[1])], np.int32)
    # cv2.polylines(frame, [left_eye_region], True, (0,0,255), 2)

    return eye_region

def get_eye_gaussed_thresholded_frame(left_eye_region, frame, threshold_level=45):
    min_x = np.min(left_eye_region[:,0])
    max_x = np.max(left_eye_region[:,0])
    min_y = np.min(left_eye_region[:,1])
    max_y = np.max(left_eye_region[:,1])

    eye_frame = frame[min_y:max_y, min_x:max_x]
    eye_frame = cv2.resize(eye_frame, None, fx=5, fy=5)
    gray_eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Eye frame", gray_eye_frame)

    gray_eye_frame = cv2.GaussianBlur(gray_eye_frame, (7,7), 0) # kernel params must be odd numbers
    _, threshold_gray_eye_frame = cv2.threshold(gray_eye_frame, threshold_level, 255, cv2.THRESH_BINARY_INV)

    return threshold_gray_eye_frame

def split_eye_frame_into_three_parts(threshold_gray_eye_frame):
    height, width = threshold_gray_eye_frame.shape
    third_of_width = int(width/3)

    left_part = threshold_gray_eye_frame[0:height, 2*third_of_width:width] # because of mirror effect this is left
    center_part = threshold_gray_eye_frame[0:height, third_of_width:2*third_of_width]
    right_part = threshold_gray_eye_frame[0:height, 0:third_of_width] # because of mirror effect this is right

    return left_part, center_part, right_part

def show_eye_parts_frames(left_part, center_part, right_part):
    cv2.imshow("Left eye part frame", left_part)
    cv2.imshow("Center eye part frame", center_part)
    cv2.imshow("Right eye part frame", right_part)

def count_white_surface(left_part, center_part, right_part):
    white_on_left_part = cv2.countNonZero(left_part)
    white_on_center_part = cv2.countNonZero(center_part)
    white_on_right_part = cv2.countNonZero(right_part)

    return white_on_left_part, white_on_center_part, white_on_right_part

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
    eye = get_eye(landmarks, left_point_num=36, right_point_num=39, top_left_point_num=37, top_right_point_num=38, bottom_left_point_num=41, bottom_right_point_num=40)

    return eye

def get_right_eye(landmarks):
    eye = get_eye(landmarks, left_point_num=42, right_point_num=45, top_left_point_num=43, top_right_point_num=44, bottom_left_point_num=47, bottom_right_point_num=46)

    return eye 

def get_eye(landmarks, left_point_num, right_point_num, top_left_point_num, top_right_point_num, bottom_left_point_num, bottom_right_point_num):
    left_point = get_landmark_point(landmarks, left_point_num)
    right_point = get_landmark_point(landmarks, right_point_num)

    top_left_point = get_landmark_point(landmarks, top_left_point_num)
    top_right_point = get_landmark_point(landmarks, top_right_point_num)
    top_point = get_point_between_two_points(top_left_point, top_right_point)

    bottom_left_point = get_landmark_point(landmarks, bottom_left_point_num)
    bottom_right_point = get_landmark_point(landmarks, bottom_right_point_num)
    bottom_point = get_point_between_two_points(bottom_left_point, bottom_right_point)

    eye = Eye(left_point, right_point, top_point, bottom_point, top_left_point, top_right_point, bottom_left_point, bottom_right_point)
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
    is_left_eye_closed, has_blinked = determine_if_eye_has_blinked(is_left_eye_closed, left_eye_ratio, has_blinked, was_blink_long_ago)
    is_right_eye_closed, has_blinked = determine_if_eye_has_blinked(is_right_eye_closed, right_eye_ratio, has_blinked, was_blink_long_ago)

    if has_blinked:
        num_of_frames_after_blink = 0

    return is_left_eye_closed, is_right_eye_closed, num_of_frames_after_blink, has_blinked

def determine_if_eye_has_blinked(is_eye_closed, eye_ratio, has_blinked, was_blink_long_ago):
    if eye_ratio > CLOSED_EYE_RATIO:
        if (was_blink_long_ago and is_eye_closed != True and (not has_blinked)):
            has_blinked = True
            # print("\n\n\n\nBLINKED\n\n\n\n")
            num_of_frames_after_blink = 0
        is_eye_closed = True
    elif eye_ratio < OPEN_EYE_RATIO:
        is_eye_closed = False

    return is_eye_closed, has_blinked

def determine_gaze_direction(gaze_direction, white_on_left_part, white_on_center_part, white_on_right_part, num_of_frames_after_gaze_direction_change, num_of_frames_after_blink):
    increase_left = 0
    increase_right = 0
    
    if (num_of_frames_after_gaze_direction_change > 5 and num_of_frames_after_blink > 20):
        gaze_direction_based_on_white = "CENTER"

        if (white_on_right_part > white_on_center_part):
            gaze_direction_based_on_white = "RIGHT"
        elif (white_on_left_part > 6*int(white_on_center_part / 10)): 
            gaze_direction_based_on_white = "LEFT"

        if (gaze_direction != gaze_direction_based_on_white):
            if (gaze_direction_based_on_white == "LEFT"):
                increase_left = 1
            elif (gaze_direction_based_on_white == "RIGHT"):
                increase_right = 1
            gaze_direction = gaze_direction_based_on_white
            num_of_frames_after_gaze_direction_change = 0

    return gaze_direction, increase_left, increase_right, num_of_frames_after_gaze_direction_change

def test():
    folder = "eye_tracker_test_data"
    video_names = os.listdir(folder)

    sum_of_errors = 0
    num_of_videos = len(video_names)

    print("naziv\t\ttacno\tdobijeno")
    for video_name in video_names:
        errors = 0
        blinks, lefts, rights = parse_video_stats(video_name)
        counted_blinks, counted_lefts, counted_rights = analyse_video(folder + "/" + video_name)
        errors = abs(counted_blinks - blinks) + abs(counted_lefts - lefts) + abs(counted_rights - rights)
        print("\n" + video_name)
        print("\tBLINKS:\t" + str(blinks) + "\t" + str(counted_blinks))
        print("\tLEFTS:\t" + str(lefts) + "\t" + str(counted_lefts))
        print("\tRIGHTS:\t" + str(rights) + "\t" + str(counted_rights))
        sum_of_errors += errors

    mae = sum_of_errors / num_of_videos
    print("\nMAE: " + str(mae))

def parse_video_stats(filename:str):
    stats = filename.split(".")[0]
    splitted_stats = stats.split("_")
    blinks = int(splitted_stats[0])
    lefts = int(splitted_stats[1])
    rights = int(splitted_stats[2])

    return blinks, lefts, rights

def analyse_video(video_path):
    # ucitavanje videa
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num) # indeksiranje frejmova
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_FILENAME)
    is_left_eye_closed, is_right_eye_closed, are_eyes_closed, num_of_frames_after_blink, num_of_blinks = setup_eyes_vars()
    gaze_direction, num_of_frames_after_gaze_direction_change, gazes_to_left, gazes_to_right = setup_gaze_vars()

    while True:
        frame_num += 1
        grabbed, frame = cap.read()

        # ako frejm nije zahvacen
        # znaci da se stiglo do kraja videa
        if not grabbed:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        num_of_frames_after_blink += 1
        num_of_frames_after_gaze_direction_change += 1

        faces = detector(gray)
        for face in faces:
            # add_face_rectangle(frame, face)
            landmarks = predictor(gray, face)
            left_eye = get_left_eye(landmarks)
            right_eye = get_right_eye(landmarks)
            # draw_lines_for_eye(frame, left_eye)
            # draw_lines_for_eye(frame, right_eye)
            left_eye_ratio = get_eye_distance_ration(left_eye)
            right_eye_ratio = get_eye_distance_ration(right_eye)
            # print(left_eye_ratio, right_eye_ratio)

            is_left_eye_closed, is_right_eye_closed, num_of_frames_after_blink, has_blinked = determine_if_blinked(is_left_eye_closed, left_eye_ratio, is_right_eye_closed, right_eye_ratio, num_of_frames_after_blink)
            
            if (has_blinked):
                num_of_blinks += 1
            else:
                left_eye_region = get_only_eye_region(left_eye)
                threshold_gray_eye_frame = get_eye_gaussed_thresholded_frame(left_eye_region, frame)
                # cv2.imshow("Eye frame", threshold_gray_eye_frame)

                left_part, center_part, right_part = split_eye_frame_into_three_parts(threshold_gray_eye_frame)
                # show_eye_parts_frames(left_part, center_part, right_part)
  
                white_on_left_part, white_on_center_part, white_on_right_part = count_white_surface(left_part, center_part, right_part)

                # print(white_on_left_part, white_on_center_part, white_on_right_part)
                gaze_direction, increase_left, increase_right, num_of_frames_after_gaze_direction_change = determine_gaze_direction(gaze_direction, white_on_left_part, white_on_center_part, white_on_right_part, num_of_frames_after_gaze_direction_change, num_of_frames_after_blink)
                gazes_to_left += increase_left
                gazes_to_right += increase_right

            # print("BLINKS: " + str(num_of_blinks) + "   " + "LEFT: " + str(gazes_to_left) + "   " + "RIGHT: " + str(gazes_to_right))
        # cv2.imshow("Frame", frame)
        

    print(num_of_blinks, gazes_to_left, gazes_to_right)

    return num_of_blinks, gazes_to_left, gazes_to_right

if __name__ == "__main__":
    # real_time()
    invalid_choice = True
    invalid_choice_threshold = True

    while invalid_choice:
        choice = input("<test/realtime>: ")
        
        if choice in ["test", "realtime"]:
            invalid_choice = False

        if choice == "test":
            test()
        elif choice == "realtime":
            while invalid_choice_threshold:
                threshold_level = input("threshold_level <1-254>: ")

                try:
                    threshold_level = int(threshold_level)

                    if (threshold_level >= 1 and threshold_level <= 254):
                        invalid_choice_threshold = False
                        real_time(threshold_level)
                except:
                    print("from 1 to 254 !")