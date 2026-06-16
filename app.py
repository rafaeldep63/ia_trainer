import streamlit as st
import tempfile
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time  # <--- IMPORTANTE: Adicionado para controlar o fluxo de renderização

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
    ["Rosca Direta", "Agachamento", "Flexão de Braço"]
)

if video_file is not None:
    # SALVAR VÍDEO TEMPORÁRIO
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name
    tfile.close() 

    # CONFIGURAÇÃO MEDIAPIPE
    MODEL_PATH = "pose_landmarker_full.task"

    if not os.path.exists(MODEL_PATH):
        st.error(f"Erro: O arquivo {MODEL_PATH} não foi encontrado na pasta!")
        st.stop()

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    contador = 0
    estagio = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        results = pose_landmarker.detect_for_video(mp_frame, timestamp)

        if results.pose_landmarks:
            pontos_corpo = results.pose_landmarks[0]

            # DESENHAR PONTOS
            for landmark in pontos_corpo:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # CONFIGURAÇÃO DE LIMITES POR EXERCÍCIO
            if exercicio == "Rosca Direta":
                p1, p2, p3 = pontos_corpo[11], pontos_corpo[13], pontos_corpo[15] 
                limite_extensao, limite_flexao = 150, 50  # Ajustado para maior tolerância
            elif exercicio == "Agachamento":
                p1, p2, p3 = pontos_corpo[23], pontos_corpo[25], pontos_corpo[27] 
                limite_extensao, limite_flexao = 160, 95
            elif exercicio == "Flexão de Braço":
                p1, p2, p3 = pontos_corpo[11], pontos_corpo[13], pontos_corpo[15] 
                limite_extensao, limite_flexao = 160, 95

            # Cálculo do Ângulo
            x1, y1 = p1.x * w, p1.y * h
            x2, y2 = p2.x * w, p2.y * h
            x3, y3 = p3.x * w, p3.y * h
            
            angulo = np.abs(np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)))
            if angulo > 180: angulo = 360 - angulo

            # LÓGICA DE CONTAGEM CORRIGIDA
            if exercicio == "Agachamento" or exercicio == "Flexão de Braço":
                # Começa em cima (braço/perna esticada), desce (flexiona), e conta ao subir de novo
                if angulo > limite_extensao: 
                    estagio = "esticado"
                if angulo < limite_flexao and estagio == "esticado":
                    estagio = "flexionado"
                if angulo > limite_extensao and estagio == "flexionado":
                    estagio = "esticado"
                    contador += 1
            elif exercicio == "Rosca Direta":
                # Começa embaixo (braço esticado), sobe (flexiona) e conta ao descer de novo
                if angulo > limite_extensao: 
                    estagio = "baixo"
                if angulo < limite_flexao and estagio == "baixo":
                    estagio = "cima"
                if angulo > limite_extensao and estagio == "cima":
                    estagio = "baixo"
                    contador += 1

            # Renderiza o ângulo atual na tela do vídeo para ajudar no debug
            cv2.putText(frame, f"Angulo: {int(angulo)}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # === EXIBIÇÃO CORRETA NO STREAMLIT ===
        frame_render = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder_video.image(frame_render, channels="RGB", use_container_width=True)
        placeholder_contador.metric("Repetições", contador)
        
        # Pequena pausa necessária para o Streamlit processar a imagem sem congelar
        time.sleep(0.01)

    cap.release()
    pose_landmarker.close()
    
    # Verifica se o arquivo ainda existe antes de deletar para evitar erros fatais
    if os.path.exists(video_path):
        os.unlink(video_path) 
