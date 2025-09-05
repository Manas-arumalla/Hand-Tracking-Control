import cv2
import mediapipe as mp

# Initialize webcam and MediaPipe Hands
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Tip landmark indices for each finger
finger_tips_ids = [4, 8, 12, 16, 20]
finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            lowered_fingers = []

            if lmList:
                # Thumb (special case: x-axis comparison due to hand orientation)
                if lmList[4][1] < lmList[3][1]:  # Right hand
                    lowered_fingers.append("Thumb")

                # Other fingers (y-axis comparison)
                for i in range(1, 5):  # Index to Pinky
                    tip_id = finger_tips_ids[i]
                    if lmList[tip_id][2] > lmList[tip_id - 2][2]:
                        lowered_fingers.append(finger_names[i])

                if lowered_fingers:
                    print("Lowered Fingers:", ", ".join(lowered_fingers))
                else:
                    print("All fingers are up.")

            # Draw landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
