import cv2
import pyautogui
import time
import mediapipe as mp
import webbrowser
import threading

# Links to open
link_1 = "https://mail.google.com/mail/u/0/#inbox"
link_2 = "https://www.youtube.com/"

# Initialize MediaPipe Hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Function to display prompt without blocking
def show_prompt():
    pyautogui.alert('Show your hand to the camera')

# Display prompt in a separate thread
thread = threading.Thread(target=show_prompt)
thread.start()

# Wait for 2 seconds
time.sleep(2)

hand_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and detect hands
    result = hands.process(rgb_frame)
    
    # Check if a hand is detected
    if result.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(frame, "Hand detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Show your hand", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Hand Detection', frame)
    
    if hand_detected:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

# If a hand was detected, open the web pages
if hand_detected:
    webbrowser.open(link_1)
    webbrowser.open(link_2)
else:
    pyautogui.alert('No hand detected')
