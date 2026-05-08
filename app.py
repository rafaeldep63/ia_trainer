
import streamlit as st
import tempfile
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



st.title('Meu Dashboard de Treino com I.A. 🦾')
st.write('Este dashboard utiliza a biblioteca MediaPipe para analisar os movimentos do corpo durante o treino. Ele detecta os pontos-chave do corpo e calcula os ângulos das articulações para contar as repetições dos exercícios.')

col1, col2 = st.columns([3,1])
with col1:
    st.header('Análise de Movimento')
    st.write('Aqui você pode carregar um vídeo do seu treino e analisar os movimentos do corpo. O sistema detecta os pontos-chave do corpo e calcula os ângulos das articulações para contar as repetições dos exercícios.')
    placeholder_video = st.empty()

with col2:
    st.header('Contador de Repetições')
    st.write('O contador de repetições é atualizado em tempo real com base nos movimentos detectados. Ele conta as repetições dos exercícios com base nos ângulos das articulações e na posição do corpo.')
    placeholder_contador = st.empty()


video_file = st.sidebar.file_uploader('Carregue um vídeo do seu treino', type=['mp4', 'avi', 'mov'])

exercicio = st.sidebar.selectbox('Selecione o exercício:', ['Rosca Direta', 'Agachamento', 'Flexão de Braço'])
st.sidebar.write(f'Modo {exercicio} ativo')

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    MODEL_PATH = 'pose_landmarker_full.task'
    options = PoseLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_landmarker = PoseLandmarker.create_from_options(options)

    cap = cv.VideoCapture(video_path)
    contador = 0
    estagio = None
    tempo_manual = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            st.warning('Erro ao ler o vídeo ou frame vazio')
            break
        
        tempo_manual += 33
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_frame = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
        tempo_manual = int(cap.get(cv.CAP_PROP_POS_MSEC))
        results = pose_landmarker.detect_for_video(mp_frame, tempo_manual)

        if results.pose_landmarks: 
            pontos_corpo = results.pose_landmarks[0]
            h, w, _ = frame.shape
            
            if exercicio == 'Rosca Direta':
                
                elbow_1 = pontos_corpo[13]  # cotovelo esquerdo
                cx, cy = int(elbow_1.x * w), int(elbow_1.y * h)
                elbow_2 = pontos_corpo[11]  # ombro esquerdo
                cx2, cy2 = int(elbow_2.x * w), int(elbow_2.y * h)
                elbow_3 = pontos_corpo[15]  # punho esquerdo
                cx3, cy3 = int(elbow_3.x * w), int(elbow_3.y * h)
                
                if cx < 0: cx = 0
                if cy < 0: cy = 0
                if cx2 < 0: cx2 = 0
                if cy2 < 0: cy2 = 0
                if cx3 < 0: cx3 = 0
                if cy3 < 0: cy3 = 0

                # Calcular ângulo para rosca direta (ombro-cotovelo-punho)
                angulo = np.arctan2(cy - cy2, cx - cx2) - np.arctan2(cy3 - cy2, cx3 - cx2)
                angulo_graus = np.abs(angulo*180.0/np.pi)
                print(f'ângulo rosca: {int(angulo_graus)}°')
                
                if angulo_graus > 180:
                    angulo_graus = 360 - angulo_graus
                    
                if angulo_graus > 110:  # braço estendido
                    estagio = 'cima'
                    
                if angulo_graus < 30 and estagio == 'cima':  # braço flexionado
                    contador += 1
                    estagio = 'baixo'
                  

            elif exercicio == 'Agachamento':
                pontos_corpo = results.pose_landmarks[0]
                hip = pontos_corpo[23]  # quadril esquerdo
                cx, cy = int(hip.x * w), int(hip.y * h)
                knee = pontos_corpo[25]  # joelho esquerdo
                cx2, cy2 = int(knee.x * w), int(knee.y * h)
                ankle = pontos_corpo[27]  # tornozelo esquerdo
                cx3, cy3 = int(ankle.x * w), int(ankle.y * h)
                
                if cx < 0: cx = 0
                if cy < 0: cy = 0
                if cx2 < 0: cx2 = 0
                if cy2 < 0: cy2 = 0
                if cx3 < 0: cx3 = 0
                if cy3 < 0: cy3 = 0

                # Calcular ângulo para agachamento (quadril-joelho-tornozelo)
                angulo = np.arctan2(cy - cy2, cx - cx2) - np.arctan2(cy3 - cy2, cx3 - cx2)
                angulo_graus = np.abs(angulo*180.0/np.pi)
                print(f'ângulo agachamento: {int(angulo_graus)}°')
                
                if angulo_graus > 180:
                    angulo_graus = 360 - angulo_graus
                    
                if angulo_graus > 160:  # em pé
                    estagio = 'cima'
                    
                if angulo_graus < 90 and estagio == 'cima':  # agachado
                    estagio = 'baixo'
                    contador += 1

            elif exercicio == 'Flexão de Braço':
                pontos_corpo = results.pose_landmarks[0]
                elbow_1 = pontos_corpo[13]
                cx, cy = int(elbow_1.x * w), int(elbow_1.y * h)
                elbow_2 = pontos_corpo[11]
                cx2, cy2 = int(elbow_2.x * w), int(elbow_2.y * h)
                elbow_3 = pontos_corpo[15]
                cx3, cy3 = int(elbow_3.x * w), int(elbow_3.y * h)
                elbow_4 = pontos_corpo[14]
                cx4, cy4 = int(elbow_4.x * w), int(elbow_4.y * h)
                elbow_5 = pontos_corpo[12]
                cx5, cy5 = int(elbow_5.x * w), int(elbow_5.y * h)
                elbow_6 = pontos_corpo[16]
                cx6, cy6 = int(elbow_6.x * w), int(elbow_6.y * h)
                if cx < 0: cx = 0
                if cy < 0: cy = 0
                if cx2 < 0: cx2 = 0
                if cy2 < 0: cy2 = 0
                if cx3 < 0: cx3 = 0
                if cy3 < 0: cy3 = 0
                if cx4 < 0: cx4 = 0
                if cy4 < 0: cy4 = 0
                if cx5 < 0: cx5 = 0
                if cy5 < 0: cy5 = 0
                if cx6 < 0: cx6 = 0
                if cy6 < 0: cy6 = 0

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
            print(f'Boa repetição: {contador}')
            st.write(f'Boa repetição: {contador}')
            for pose_landmarks in results.pose_landmarks:
                drawing_utils.draw_landmarks(
                    frame,
                    pose_landmarks,
                    PoseLandmarksConnections.POSE_LANDMARKS,
                    drawing_utils.DrawingSpec(color=drawing_utils.RED_COLOR, thickness=2, circle_radius=2),
                    drawing_utils.DrawingSpec(color=drawing_utils.GREEN_COLOR, thickness=2),
                )
            else:
        
              frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
              placeholder_video.image(frame_rgb, channels="RGB")

            placeholder_contador.metric(label='Contador de Repetições:' , value=contador)

else:
    st.info('👆 Carregue um vídeo do seu treino na barra lateral para começar!')