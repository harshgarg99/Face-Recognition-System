import sys
import pygame
from Button import Button
import cv2
import numpy as np
import mtcnn
from architecture import InceptionResNetV2
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
import pickle

# ------------------------------- CONFIG ----------------------------------

# Minimum confidence for MTCNN to accept face detection
confidence_t = 0.95

# 🔥 STRICT RECOGNITION THRESHOLD — LOWER = safer + more unknown detection
# Cosine distance threshold: typical good values (0.35–0.55)
recognition_t = 0.45

# 🔥 Require clear separation between best and second-best match
# Helps prevent strangers matching someone with close encoding
min_diff_best_second = 0.15

required_size = (160, 160)

# ------------------------------- FACE UTILS ----------------------------------


def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)


def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    with open(path, "rb") as f:
        encoding_dict = pickle.load(f)
    return encoding_dict


# ------------------------------- DETECTION + RECOGNITION ----------------------------------


def detect(img, detector, encoder, encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)

    for res in results:
        if res["confidence"] < confidence_t:
            continue

        face, pt_1, pt_2 = get_face(img_rgb, res["box"])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]

        # ---------------------- SAFER MATCHING -----------------------
        best_name = "unknown"
        best_dist = float("inf")
        second_best = float("inf")

        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)

            if dist < best_dist:
                second_best = best_dist
                best_dist = dist
                best_name = db_name
            elif dist < second_best:
                second_best = dist

        # Compare with thresholds
        is_good_distance = best_dist < recognition_t
        margin = second_best - best_dist
        is_clear_margin = margin > min_diff_best_second or second_best == float("inf")

        if is_good_distance and is_clear_margin:
            name = best_name
        else:
            name = "unknown"

        # ---------------------- DRAW RESULTS -----------------------
        if name == "unknown":
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(
                img, name, (pt_1[0], pt_1[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
            )
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(
                img, f"{name}__{best_dist:.2f}",
                (pt_1[0], pt_1[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
            )

    return img


# ------------------------------- CAMERA LOOP ----------------------------------


def main_program():
    print("🔄 Loading FaceNet model...")
    face_encoder = InceptionResNetV2()
    face_encoder.load_weights("facenet_keras_weights.h5")

    print("🔄 Loading encodings...")
    encoding_dict = load_pickle("encodings/encodings.pkl")

    print("🔄 Initializing MTCNN...")
    face_detector = mtcnn.MTCNN()

    print("🎥 Opening Webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ERROR: Webcam not accessible!")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Could not read frame from camera!")
            break

        frame = detect(frame, face_detector, face_encoder, encoding_dict)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------------------- PYGAME UI ----------------------------------


def getFont(size):
    return pygame.font.Font("assets/abg.ttf", size)


def mainMenu():
    pygame.init()

    WIDTH, HEIGHT = 1280, 720
    Screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Face Recognition System")

    BG = pygame.image.load("assets/background.jpg")

    while True:
        Screen.blit(BG, (0, 0))
        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = getFont(60).render("Face Recognition with MTCNN + FaceNet", True, "#ffffff")
        MENU_RECT = MENU_TEXT.get_rect(center=(640, 150))
        Screen.blit(MENU_TEXT, MENU_RECT)

        START_BUTTON = Button(
            image=pygame.image.load("assets/Play Rect.png"),
            pos=(640, 300),
            text_input="START",
            font=getFont(50),
            base_color="Black",
            hovering_color="White",
        )

        QUIT_BUTTON = Button(
            image=pygame.image.load("assets/Quit Rect.png"),
            pos=(640, 550),
            text_input="QUIT",
            font=getFont(50),
            base_color="Black",
            hovering_color="White",
        )

        # Button animations
        for button in [START_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(Screen)

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:

                if START_BUTTON.checkForInput(MENU_MOUSE_POS):
                    main_program()

                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()


# ------------------------------- RUN PROGRAM ----------------------------------

if __name__ == "__main__":
    mainMenu()
