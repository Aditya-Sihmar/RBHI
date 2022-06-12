from django.shortcuts import render
from django.http import JsonResponse
from .models import Class1
from .serializer import CourseSerializer
import math
import numpy as np
import cv2
import threading
import mediapipe as mp


# Create your views here.

def getdata(request):
    return JsonResponse({'data':dat_fun()})


n1 = 0
def dat_fun():
    global n1
    global position
    n1 += 1
    if n1 == 1:
        thread = threading.Thread(target=webcam_start, name="AI Model")
        thread.start()
    else:
        pass
    return [n, position]

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

n = 0
# writer = 1
position = {}
# writer = None   


def angle_3d(a1, b1, c1, a2, b2, c2):
     
    d = ( a1 * a2 + b1 * b2 + c1 * c2 )
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    A = math.degrees(math.acos(d))
    return A

def angle_2d(m1, m2):
    d = np.abs((m1-m2)/(1+m1*m2))
    A = math.degrees(math.atan(d))
    return A

def slope(xs, ys):
    return (ys[0]-ys[1])/(xs[0]-xs[1])

def distance(p1, p2):
    dist = np.linalg.norm(p1 - p2)
    return dist

def determine(lst):
    points = []
    for pt in lst:
        points.append(np.array([pt.x, pt.y, pt.z]))

    # m1 = slope(x[0:2], y[0:2])
    # m2 = slope(x[1:], y[1:])
    # angle = angle_2d(m1,m2)
    # if angle > 2:
    #     print(f'{angle} close')
    # else:
    #     print(f'{angle} open')
    d1 = distance(points[0],points[1])+distance(points[1], points[2]) + distance(points[2],points[3])
    d2 = distance(points[0], points[3])
    if np.abs(d1 - d2) >= 0.015:
        return 0
    else:
        return 1

def webcam_start():    
    # For webcam input:
    global position
    global n
    print("Entered")
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        # model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                hand_name = [handedness.classification[0].label for handedness in results.multi_handedness]  
                for hand_landmarks in results.multi_hand_landmarks:
                    # print(hand_landmarks.landmark[5:9])
                    nme = hand_name.pop(0)
                    
                    print(f'{nme} : {len(hand_landmarks.landmark)}')
                    if nme == 'Right':
            ####################################################################################################################################################
                        position = {'index': determine(hand_landmarks.landmark[5:9]),
                                    'middle': determine(hand_landmarks.landmark[9:13]),
                                    'ring': determine(hand_landmarks.landmark[13:17]),
                                    'pinky': determine(hand_landmarks.landmark[17:])}
            ####################################################################################################################################################
                        print(position)
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS)
                            # mp_drawing_styles.get_default_hand_landmarks_style(),
                            # mp_drawing_styles.get_default_hand_connections_style())
                    
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            # cv2.imwrite(f'outputs/{n}.jpeg', image)
            n += 1
            # writer.write(cv2.flip(image,1))
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cap.release()
    
