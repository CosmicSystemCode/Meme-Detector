import cv2
import mediapipe as mp

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Carrega a imagem
euvou_img = cv2.imread("macaco.jpeg")

# Inicia a captura da câmera
cap = cv2.VideoCapture(0)

def dedo_levantado(landmarks):
    dedos = []

    # Posições de referência (índices dos landmarks do mediapipe)
    dedos_tips = [4, 8, 12, 16, 20]

    # Polegar (comparando eixo X)
    dedos.append(landmarks[dedos_tips[0]].x < landmarks[dedos_tips[0] - 2].x)

    # Outros dedos (comparando eixo Y)
    for tip in dedos_tips[1:]:
        dedos.append(landmarks[tip].y < landmarks[tip - 2].y)

    return dedos

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            dedos = dedo_levantado(hand_landmarks.landmark)

            # Se apenas o indicador estiver levantado
            if dedos[1] and not any(dedos[i] for i in [0, 2, 3, 4]):
                # Redimensiona e mostra a imagem "euvou"
                img_resized = cv2.resize(euvou_img, (300, 200))
                h, w, _ = img_resized.shape
                frame[0:h, 0:w] = img_resized

    cv2.imshow("Detector de Memes - EU VOU", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Pressione ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
