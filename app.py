import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="Dashboard de Treino IA", layout="wide")

st.title("Meu Dashboard de Treino com I.A. 🦾")

st.write(
    "Este dashboard utiliza MediaPipe para analisar movimentos "
    "durante exercícios físicos."
)

col1, col2 = st.columns([3, 1])

with col1:
    st.header("Análise de Movimento")
    placeholder_video = st.empty()

with col2:
    st.header("Contador de Repetições")
    placeholder_contador = st.empty()

video_file = st.sidebar.file_uploader(
    "Carregue um vídeo",
    type=["mp4", "avi", "mov"]
)

exercicio = st.sidebar.selectbox(
    "Selecione o exercício",
    ["Rosca Direta", "Agachamento", "Flexão de Braço"]
)

if video_file is not None:

    # Salvar vídeo temporário
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    video_path = tfile.name

    # Modelo do MediaPipe
    MODEL_PATH = "pose_landmarker_full.task"

    # Configurar MediaPipe
    BaseOptions = python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose_landmarker = PoseLandmarker.create_from_options(
        options
    )

    # Abrir vídeo
    cap = cv2.VideoCapture(video_path)

    # Melhorar estabilidade
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    contador = 0
    estagio = None
    frame_skip = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Reduzir processamento
        frame_skip += 1

        if frame_skip % 5 != 0:
            continue

        # Converter para RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Criar imagem MediaPipe
        mp_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = int(
            cap.get(cv2.CAP_PROP_POS_MSEC)
        )

        # Detectar pose
        results = pose_landmarker.detect_for_video(
            mp_frame,
            timestamp
        )

        if results.pose_landmarks:

            pontos_corpo = results.pose_landmarks[0]

            h, w, _ = frame.shape

            # ===================================
            # ROSCA DIRETA
            # ===================================
            if exercicio == "Rosca Direta":

                shoulder = pontos_corpo[11]
                elbow = pontos_corpo[13]
                wrist = pontos_corpo[15]

                x1, y1 = int(shoulder.x * w), int(shoulder.y * h)
                x2, y2 = int(elbow.x * w), int(elbow.y * h)
                x3, y3 = int(wrist.x * w), int(wrist.y * h)

                angulo = np.arctan2(
                    y3 - y2,
                    x3 - x2
                ) - np.arctan2(
                    y1 - y2,
                    x1 - x2
                )

                angulo = np.abs(angulo * 180 / np.pi)

                if angulo > 180:
                    angulo = 360 - angulo

                if angulo > 110:
                    estagio = "baixo"

                if angulo < 40 and estagio == "baixo":
                    estagio = "cima"
                    contador += 1

            # ===================================
            # AGACHAMENTO
            # ===================================
            elif exercicio == "Agachamento":

                hip = pontos_corpo[23]
                knee = pontos_corpo[25]
                ankle = pontos_corpo[27]

                x1, y1 = int(hip.x * w), int(hip.y * h)
                x2, y2 = int(knee.x * w), int(knee.y * h)
                x3, y3 = int(ankle.x * w), int(ankle.y * h)

                angulo = np.arctan2(
                    y3 - y2,
                    x3 - x2
                ) - np.arctan2(
                    y1 - y2,
                    x1 - x2
                )

                angulo = np.abs(angulo * 180 / np.pi)

                if angulo > 180:
                    angulo = 360 - angulo

                if angulo > 160:
                    estagio = "cima"

                if angulo < 90 and estagio == "cima":
                    estagio = "baixo"
                    contador += 1

            # ===================================
            # FLEXÃO
            # ===================================
            elif exercicio == "Flexão de Braço":

                shoulder = pontos_corpo[11]
                elbow = pontos_corpo[13]
                wrist = pontos_corpo[15]

                x1, y1 = int(shoulder.x * w), int(shoulder.y * h)
                x2, y2 = int(elbow.x * w), int(elbow.y * h)
                x3, y3 = int(wrist.x * w), int(wrist.y * h)

                angulo = np.arctan2(
                    y3 - y2,
                    x3 - x2
                ) - np.arctan2(
                    y1 - y2,
                    x1 - x2
                )

                angulo = np.abs(angulo * 180 / np.pi)

                if angulo > 180:
                    angulo = 360 - angulo

                if angulo > 110:
                    estagio = "baixo"

                if angulo < 40 and estagio == "baixo":
                    estagio = "cima"
                    contador += 1

            # Desenhar pose
            for pose_landmarks in results.pose_landmarks:

                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )

        # Mostrar vídeo
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        if frame_skip % 15 == 0:
            placeholder_video.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

        # Mostrar contador
        placeholder_contador.metric(
            "Repetições",
            contador
        )

    cap.release()

else:
    st.info(
        "👆 Faça upload de um vídeo para começar!"
    )