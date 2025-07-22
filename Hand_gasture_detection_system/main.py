import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip for natural interaction
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = hand_landmarks.landmark

            # Index finger tip (landmark 8)
            x = int(lm_list[8].x * img.shape[1])
            y = int(lm_list[8].y * img.shape[0])

            # Convert to screen coordinates
            screen_x = int(lm_list[8].x * screen_width)
            screen_y = int(lm_list[8].y * screen_height)

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Optional: Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()