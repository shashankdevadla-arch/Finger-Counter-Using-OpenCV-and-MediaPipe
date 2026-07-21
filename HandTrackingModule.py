import cv2
import mediapipe as mp
import math

class HandDetector:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
        self.results = None
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results and self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape

            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        self.lmList = lmList
        return lmList

    def fingersUp(self):
        fingers = []

        if len(self.lmList) == 0:
            return fingers

        # Hand type
        if self.results and self.results.multi_handedness:
            handType = self.results.multi_handedness[0].classification[0].label
        else:
            handType = "Right"

        # Thumb
        # Thumb (improved)
        thumb_tip = self.lmList[4]
        thumb_ip = self.lmList[3]
        wrist = self.lmList[0]
        
        thumb_dist = ((thumb_tip[1] - wrist[1]) ** 2 + (thumb_tip[2] - wrist[2]) ** 2)
        ip_dist = ((thumb_ip[1] - wrist[1]) ** 2 + (thumb_ip[2] - wrist[2]) ** 2)
        
        if thumb_dist > ip_dist:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

 


    






