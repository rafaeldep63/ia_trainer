import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(
    page_title="Dashboard IA",
    layout="wide"
)

st.title("Meu Dashboard de Treino com I.A. 🦾")

st.write(
    "Sistema de análise corporal com MediaPipe "
    "e visão computacional."
)

col1, col2 = st.columns([3, 1])

with col1:
    st.header("Análise de Movimento")
    placeholder_video = st.empty()

with col2:
    st.header("Contador")
    placeholder_contador = st.empty()

video_file = st.sidebar.file_uploader(
    "Carregue um vídeo",
    type=["mp4", "avi", "mov"]
)

exercicio = st.sidebar.selectbox(
    "Selecione o exercício",
    [
        "Rosca Direta",
        "Agachamento",
        "Flexão de Braço"
    ]
)

if video_file is not None:

    # =========================
    # SALVAR VÍDEO
    # =========================
    tfile = tempfile.NamedTemporaryFile(
        delete=False
    )

    tfile.write(video_file.read())

    video_path = tfile.name

    # =========================
    # MEDIAPIPE
    # =========================
    MODEL_PATH = "pose_landmarker_full.task"

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

    # =========================
    # VÍDEO
    # =========================
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    contador = 0
    estagio = None
    frame_skip = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_skip += 1

        # Reduz lag
        if frame_skip % 5 != 0:
            continue

        # =========================
        # RGB
        # =========================
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp = int(
            cap.get(cv2.CAP_PROP_POS_MSEC)
        )

        # =========================
        # DETECÇÃO
        # =========================
        results = pose_landmarker.detect_for_video(
            mp_frame,
            timestamp
        )

        if results.pose_landmarks:

            pontos_corpo = results.pose_landmarks[0]

            h, w, _ = frame.shape

            # =========================
            # DESENHAR PONTOS
            # =========================
            for pose_landmarks in results.pose_landmarks:

                for landmark in pose_landmarks:

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

            # =========================
            # ROSCA DIRETA
            # =========================
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

                angulo = np.abs(
                    angulo * 180 / np.pi
                )

                if angulo > 180:
                    angulo = 360 - angulo

                if angulo > 110:
                    estagio = "baixo"

                if angulo < 40 and estagio == "baixo":
                    estagio = "cima"
                    contador += 1

            # =========================
            # AGACHAMENTO
            # =========================
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

                angulo = np.abs(
                    angulo * 180 / np.pi
                )

                if angulo > 180:
                    angulo = 360 - angulo

                if angulo > 160:
                    estagio = "cima"

                if angulo < 90 and estagio == "cima":
                    estagio = "baixo"
                    contador += 1

            # =========================
            # FLEXÃO
            # =========================
            elif exercicio == "Flexão de Braço":

                shoulder = pontos_corpo[11]
                
            if video_file is not None:
    # Correção do Tempfile: use suffix para garantir que o OpenCV entenda a extensão
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(video_file.read())
                tfile.close() # FECHE antes de abrir com o CV2

                    video_path = tfile.name
                    cap = cv2.VideoCapture(video_path)

    # ... (configuração do mediapipe igual)

                        while cap.isOpened():
                          ret, frame = cap.read()
                            if not ret:
                                  break

        # [Seu processamento de mediapipe e lógica de ângulos aqui...]

        # === A PARTE QUE FALTA: EXIBIR NO STREAMLIT ===
        
        # 1. Converte o frame desenhado para RGB
                            frame_visual = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Atualiza os placeholders que você criou no início do script
                        placeholder_video.image(frame_visual, channels="RGB")
                         placeholder_contador.subheader(f"{contador}")

cap.release()
