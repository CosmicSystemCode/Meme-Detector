import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Função que retorna um dicionário indicando quais dedos estão levantados
def finger_states(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = {}

    # Indicador
    fingers['index'] = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
    # Médio
    fingers['middle'] = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    # Anelar
    fingers['ring'] = lm[mp_hands.HandLandmark.RING_FINGER_TIP].y < lm[mp_hands.HandLandmark.RING_FINGER_PIP].y
    # Mindinho
    fingers['pinky'] = lm[mp_hands.HandLandmark.PINKY_TIP].y < lm[mp_hands.HandLandmark.PINKY_PIP].y

    # Polegar — compara eixo X (porque o polegar aponta para o lado)
    fingers['thumb'] = lm[mp_hands.HandLandmark.THUMB_TIP].x > lm[mp_hands.HandLandmark.THUMB_IP].x

    return fingers

def is_hand_open(states):
    return all(states.values())  # todos levantados

def is_hand_closed(states):
    return not any(states.values())  # nenhum levantado

def one_finger_up(states):
    return sum(states.values()) == 1

def only_index_up(states):
    return states['index'] and sum(states.values()) == 1

def only_thumb_up(states):
    return states['thumb'] and sum(states.values()) == 1

def thumb_and_index_up(states):
    return states['thumb'] and states['index'] and sum(states.values()) == 2


# Carregamento das imagens
IMAGES = {
    "duas_abertas_1": "davi.jpg",
    "duas_abertas_2": "cinema.jpg",
    "coelho": "coelho.jpg",
    "duas_fechadas": "eumato.jpg",
    "galo": "galo.jpg",
    "euvou": "euvou.jpg",
    "fazol": "fazol.jpg"
}

MEME_WIDTH, MEME_HEIGHT = 200, 150
loaded_images = {}

print("Carregando imagens...")
for key, path in IMAGES.items():
    img = cv2.imread(path)
    if img is None:
        print(f"Erro ao carregar {path}")
        continue
    loaded_images[key] = cv2.resize(img, (MEME_WIDTH, MEME_HEIGHT))
print("Imagens carregadas:", list(loaded_images.keys()))

# Webcam
cap = cv2.VideoCapture(0)
current_image = None
alternate = True 

print("Iniciando... Pressione 'q' para sair.")

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
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_states.append(finger_states(hand_landmarks))

        # Lógica principal
        if len(hand_states) == 2:
            left, right = hand_states[0], hand_states[1]

            if is_hand_open(left) and is_hand_open(right):
                current_image = loaded_images["duas_abertas_1" if alternate else "duas_abertas_2"]
                alternate = not alternate

            elif (is_hand_closed(left) and only_index_up(right)) or (is_hand_closed(right) and only_index_up(left)):
                current_image = loaded_images["coelho"]

            elif is_hand_closed(left) and is_hand_closed(right):
                current_image = loaded_images["duas_fechadas"]

            else:
                current_image = None

        elif len(hand_states) == 1:
            s = hand_states[0]

            if only_thumb_up(s):
                current_image = loaded_images["galo"]
            elif only_index_up(s):
                current_image = loaded_images["euvou"]
            elif thumb_and_index_up(s):
                current_image = loaded_images["fazol"]
            else:
                current_image = None
        else:
            current_image = None

    else:
        current_image = None

    # Exibir imagem na tela
    if current_image is not None:
        x_offset = int((frame_w - MEME_WIDTH) / 2)
        y_offset = 20
        frame[y_offset:y_offset + MEME_HEIGHT, x_offset:x_offset + MEME_WIDTH] = current_image

    cv2.imshow("Detector de Memes", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

