import cv2
import mediapipe as mp

# Inicialização do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Função para verificar se a mão está aberta (todos os dedos levantados)
def is_hand_open(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = {}
    fingers['index'] = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    fingers['middle'] = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    fingers['ring'] = lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    fingers['pinky'] = lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_PIP].y
    return all(fingers.values())

# Função que retorna o estado dos dedos
def finger_states(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = {
        'index': lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
        'middle': lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
        'ring': lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y,
        'pinky': lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_PIP].y,
        'thumb': False  # será ajustado conforme o lado
    }
    return fingers


def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    MEME_WIDTH, MEME_HEIGHT = 250, 180  # tamanho reduzido da imagem

    # Carregamento das imagens
    IMAGES = {
        "cinema": cv2.imread("cinema.jpg"),
        "coelho": cv2.imread("coelho.jpg"),
        "eumato": cv2.imread("eumato.jpg"),
        "galo": cv2.imread("galo.jpg"),
        "euvou": cv2.imread("euvou.jpg"),
        "fazol": cv2.imread("fazol.jpg"),
        "dedodomeio": cv2.imread("dedodomeio.jpeg")
    }

    # Redimensiona todas as imagens
    for key in IMAGES:
        if IMAGES[key] is not None:
            IMAGES[key] = cv2.resize(IMAGES[key], (MEME_WIDTH, MEME_HEIGHT))

    current_image = None

    print("Detector de Memes iniciado! Pressione 'q' para sair.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        frame_h, frame_w, _ = frame.shape

        if results.multi_hand_landmarks:
            hand_states = []

            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark
                label = handedness.classification[0].label  # "Left" ou "Right"
                states = finger_states(hand_landmarks)

                # Corrige o polegar dependendo do lado
                if label == 'Right':
                    states['thumb'] = lm[mp_hands.HandLandmark.THUMB_TIP].x > lm[mp_hands.HandLandmark.THUMB_IP].x
                else:
                    states['thumb'] = lm[mp_hands.HandLandmark.THUMB_TIP].x < lm[mp_hands.HandLandmark.THUMB_IP].x

                hand_states.append((states, label, hand_landmarks))

            num_hands = len(hand_states)

            # ===== LÓGICA DE DETECÇÃO =====
            if num_hands == 2:
                left, right = hand_states[0][0], hand_states[1][0]
                left_lm, right_lm = hand_states[0][2], hand_states[1][2]

                # Duas mãos abertas → cinema.jpg
                if is_hand_open(hand_states[0][2]) and is_hand_open(hand_states[1][2]):
                    current_image = IMAGES["cinema"]

                # Uma fechada + uma com indicador → coelho.jpg
                elif (not any(left.values()) and sum(right.values()) == 1 and right['index']) or \
                     (not any(right.values()) and sum(left.values()) == 1 and left['index']):
                    current_image = IMAGES["coelho"]

                # Duas fechadas → eumato.jpg
                elif not any(left.values()) and not any(right.values()):
                    current_image = IMAGES["eumato"]
                else:
                    current_image = None

            elif num_hands == 1:
                s, handedness_label, lm = hand_states[0]

                # Apenas polegar levantado → galo.jpg
                if s['thumb'] and not any([s['index'], s['middle'], s['ring'], s['pinky']]):
                    current_image = IMAGES["galo"]

                # Apenas indicador levantado → euvou.jpg
                elif s['index'] and not any([s['thumb'], s['middle'], s['ring'], s['pinky']]):
                    current_image = IMAGES["euvou"]

                # Indicador para cima e polegar para o lado → fazol.jpg (gesto de L)
                else:
                    thumb_open = (
                        (handedness_label == 'Right' and lm.landmark[mp_hands.HandLandmark.THUMB_TIP].x >
                         lm.landmark[mp_hands.HandLandmark.THUMB_IP].x) or
                        (handedness_label == 'Left' and lm.landmark[mp_hands.HandLandmark.THUMB_TIP].x <
                         lm.landmark[mp_hands.HandLandmark.THUMB_IP].x)
                    )

                    index_up = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < \
                               lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

                    other_closed = not (
                        lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y <
                        lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y or
                        lm.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y <
                        lm.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y or
                        lm.landmark[mp_hands.HandLandmark.PINKY_TIP].y <
                        lm.landmark[mp_hands.HandLandmark.PINKY_PIP].y
                    )

                    if thumb_open and index_up and other_closed:
                        current_image = IMAGES["fazol"]
                    else:
                        current_image = None
            else:
                current_image = None
        else:
            current_image = None

        # Exibir imagem no topo centralizada
        if current_image is not None:
            x_offset = (frame_w - MEME_WIDTH) // 2
            y_offset = 20
            frame[y_offset:y_offset + MEME_HEIGHT, x_offset:x_offset + MEME_WIDTH] = current_image

        cv2.imshow("Detector de Memes", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()