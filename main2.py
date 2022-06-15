from operator import ilshift
import cv2
from sklearn.metrics import euclidean_distances
import mediapipe as mp
import numpy as np
import json
import math
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def motor_angles(hand_coords):
  fingers = ['INDEX_FINGER','MIDDLE_FINGER','RING_FINGER','PINKY']
  points = ['MCP','PIP','TIP']
  angles = {}
  for hand_no in hand_coords:
    
    landmarks = hand_coords[hand_no]
    if len(landmarks) == 21:
      per_hand = {}
      hand_coords = []
      for idx , i in enumerate(fingers):
        fing_name = i
        point_vectors = []
        for j in points:
          key = f'{i}_{j}'
          point_vectors.append(landmarks[key])
        hand_coords.append(point_vectors)
        per_hand[fing_name] = cal_angle(hand_coords[idx])
      angles[hand_no] = per_hand
    print(angles)
    return angles
  






def cal_angle(coords):
  a , b, c = np.array(coords)
  vec1 = b - a
  vec2 = b - c

  dot = np.dot(vec1, vec2)
  mag1 = np.linalg.norm(vec1)
  mag2 = np.linalg.norm(vec2)
  return np.arccos((dot/(mag1*mag2)))



# For static images:
IMAGE_FILES = []
#['Data/images/1.png', 'Data/images/2.png','Data/images/3.png']
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=3,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #int(results.multi_hand)
    # Print handedness and draw hand landmarks on the image.
    #print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    n_hands = {}
    hand_no = 0
    for hand_landmarks in results.multi_hand_landmarks:
      # #print('hand_landmarks:', hand_landmarks)
      # hand_coords = {}
      # for i in range(21):    
      #     coords = [hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x * image_width, 
      #               hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y * image_height,
      #               hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z * image_width]

      #     hand_coords[mp_hands.HandLandmark(i).name] = coords

      # n_hands[hand_no] = hand_coords
      # hand_no = hand_no + 1
    
      

      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

    # motor_angles(n_hands)
    # with open('hand_coordinates.json', 'w') as f:
    #   json.dump(n_hands, f)

    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

# For webcam input:


cap = cv2.VideoCapture(0)
_, frame = cap.read()
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter("output.avi", fourcc, fps,(int(cap.get(3)), int(cap.get(4))))

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    frame = image
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape
    hand_no = 0
    n_hands = {}
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        #print('hand_landmarks:', hand_landmarks)
        hand_coords = {}
        for i in range(21):    
            coords = [hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x * image_width, 
                      hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y * image_height,
                      hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z * image_width]

            hand_coords[mp_hands.HandLandmark(i).name] = coords

        n_hands[hand_no] = hand_coords
        hand_no = hand_no + 1
        #print(n_hands.keys())
        

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

      angles = motor_angles(n_hands)
      with open('angles.json', 'w') as f:
        json.dump(angles, f)
    # Flip the image horizontally for a selfie-view display.
        #print('Frame Change')
    writer.write(image)
    cv2.imwrite(
        'reference1.png', cv2.flip(frame, 1))
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == ord('q'):

      break
cap.release()