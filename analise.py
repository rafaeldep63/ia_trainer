import cv2 as cv
import numpy as np
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    drawing_utils,
)
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python.vision.core import vision_task_running_mode

MODEL_PATH = 'pose_landmarker_full.task'

cap = cv.VideoCapture('treino.mp4')
if not cap.isOpened():
    print('Error')
    exit()

options = PoseLandmarkerOptions(
    base_options=base_options.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose_landmarker = PoseLandmarker.create_from_options(options)

frame_count = 0
contador= 0
estagio = None

tempo_manual = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    tempo_manual += 33
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_frame = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
    tempo_manual = int(cap.get(cv.CAP_PROP_POS_MSEC))
    results = pose_landmarker.detect_for_video(mp_frame, tempo_manual)

    if results.pose_landmarks:
        h, w, _ = frame.shape

        pontos_corpo= results.pose_landmarks[0]
        
        elbow_1 = pontos_corpo[13]
        cx, cy = int(elbow_1.x * w), int(elbow_1.y * h)

        elbow_2 = pontos_corpo[11]
        cx2, cy2 = int(elbow_2.x * w), int(elbow_2.y * h)

        elbow_3 = pontos_corpo[15]
        cx3, cy3 = int(elbow_3.x * w), int(elbow_3.y * h)

        if cx< 0 : cx = 0
        if cy< 0 : cy = 0
        if cx2< 0 : cx2 = 0
        if cy2< 0 : cy2 = 0
        if cx3< 0 : cx3 = 0
        if cy3< 0 : cy3 = 0

        cv.circle(frame, (cx, cy), 10, (0, 255, 0,), -1)
        print(f'cotovelo: {cx}, {cy}')
        cv.putText(frame, f'cotovelo: {cx}, {cy}', (cx, cy-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(frame, (cx2, cy2), 10, (0, 255, 0,), -1)
        print(f'ombro: {cx2}, {cy2}')
        cv.putText(frame, f'ombro: {cx2}, {cy2}', (cx2, cy2-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.circle(frame, (cx3, cy3), 10, (0, 255, 0,), -1)
        print(f'punho: {cx3}, {cy3}')
        cv.putText(frame, f'punho: {cx3}, {cy3}', (cx3, cy3-20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        
        angulo = np.arctan2(cy - cy2, cx - cx2) - np.arctan2(cy3 - cy2, cx3 - cx2)
        angulo_graus = np.abs(angulo*180.0/np.pi)
        print(f'ângulo: {int(angulo_graus)}°')
    if angulo_graus > 180:
            angulo_graus = 360 - angulo_graus
            
    if angulo_graus > 110:
        estagio = 'baixo'

    if angulo_graus < 40 and estagio == 'baixo':
                 estagio = 'cima'
                 contador += 1
    
    cv.putText(frame, f'contador: {contador}', (cx2, cy2+60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print(f'Boa repetição: {contador}')
    for pose_landmarks in results.pose_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                pose_landmarks,
                PoseLandmarksConnections.POSE_LANDMARKS,
                drawing_utils.DrawingSpec(color=drawing_utils.RED_COLOR, thickness=2, circle_radius=2),
                drawing_utils.DrawingSpec(color=drawing_utils.GREEN_COLOR, thickness=2),
            )

    cv.putText(frame, f'Frame: {frame_count}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.imshow('Pose Landmarks', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.namedWindow('Pose Landmarks', cv.WINDOW_NORMAL)
cap.release()
cv.destroyAllWindows() 