import os
import tempfile

import cv2 as cv
import numpy as np
import streamlit as st
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


def calcular_angulo(p1, p2, p3):
    """Calcula o ângulo em graus formado por três pontos (p1-p2-p3), medido em p2."""
    a = np.array([p1.x, p1.y])
    b = np.array([p2.x, p2.y])
    c = np.array([p3.x, p3.y])

    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angulo = float(np.abs(np.degrees(rad)))

    if angulo > 180.0:
        angulo = 360.0 - angulo

    return angulo


st.title('Meu Dashboard de Treino com I.A. 🦾')
st.write('Este dashboard utiliza a biblioteca MediaPipe para analisar os movimentos do corpo durante o treino. Ele detecta os pontos-chave do corpo e calcula os ângulos das articulações para contar as repetições dos exercícios.')

col1, col2 = st.columns([1, 1])
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
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name

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

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            tempo_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_frame = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
            results = pose_landmarker.detect_for_video(mp_frame, tempo_ms)

            if results.pose_landmarks:
                pontos_corpo = results.pose_landmarks[0]

                if exercicio == 'Rosca Direta':
                    # média do ângulo dos dois braços (ombro-cotovelo-punho)
                    angulo = (
                        calcular_angulo(pontos_corpo[11], pontos_corpo[13], pontos_corpo[15])
                        + calcular_angulo(pontos_corpo[12], pontos_corpo[14], pontos_corpo[16])
                    ) / 2
                    print(f'ângulo rosca: {int(angulo)}°')

                    if angulo > 110:  # braços estendidos
                        estagio = 'cima'

                    if angulo < 30 and estagio == 'cima':  # braços flexionados
                        contador += 1
                        estagio = 'baixo'

                elif exercicio == 'Agachamento':
                    # ângulo do joelho esquerdo (quadril-joelho-tornozelo)
                    angulo = calcular_angulo(pontos_corpo[23], pontos_corpo[25], pontos_corpo[27])
                    print(f'ângulo agachamento: {int(angulo)}°')

                    if angulo > 160:  # em pé
                        estagio = 'cima'

                    if angulo < 90 and estagio == 'cima':  # agachado
                        estagio = 'baixo'
                        contador += 1

                elif exercicio == 'Flexão de Braço':
                    # média do ângulo dos dois braços (ombro-cotovelo-punho)
                    angulo = (
                        calcular_angulo(pontos_corpo[11], pontos_corpo[13], pontos_corpo[15])
                        + calcular_angulo(pontos_corpo[12], pontos_corpo[14], pontos_corpo[16])
                    ) / 2
                    print(f'ângulo flexão: {int(angulo)}°')

                    if angulo > 110:  # braços estendidos
                        estagio = 'baixo'

                    if angulo < 40 and estagio == 'baixo':  # braços flexionados
                        estagio = 'cima'
                        contador += 1

                for pose_landmarks in results.pose_landmarks:
                    drawing_utils.draw_landmarks(
                        frame,
                        pose_landmarks,
                        PoseLandmarksConnections.POSE_LANDMARKS,
                        drawing_utils.DrawingSpec(color=drawing_utils.RED_COLOR, thickness=2, circle_radius=2),
                        drawing_utils.DrawingSpec(color=drawing_utils.GREEN_COLOR, thickness=2),
                    )

            placeholder_video.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB), channels='RGB')
            placeholder_contador.metric(label='Repetições', value=contador)
    finally:
        cap.release()
        pose_landmarker.close()
        try:
            os.remove(video_path)
        except OSError:
            pass

    st.success(f'Análise finalizada! Total de repetições: {contador}')
else:
    st.info('👆 Carregue um vídeo do seu treino na barra lateral para começar!')
