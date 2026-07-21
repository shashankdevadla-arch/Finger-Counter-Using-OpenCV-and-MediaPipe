from operator import index

import cv2
import time
import os


import HandTrackingModule as htm


folderPath = "C:\\Users\\nikhi\\OneDrive\\Pictures\\FingerImages"
myList = [
    f for f in os.listdir(folderPath)
    if f.endswith(".jpg") and os.path.splitext(f)[0].isdigit()
]

myList.sort(key=lambda x: int(os.path.splitext(x)[0]))

overlayList = []

for imgPath in myList:
    if imgPath.endswith(".jpg"):
        fullPath = os.path.join(folderPath, imgPath)

        image = cv2.imread(fullPath)

        if image is None:
            print("Failed to load:", imgPath)
        else:
            image = cv2.resize(image, (200, 200))
        overlayList.append(image)

print("Overlay images loaded:", len(overlayList))



detector=htm.HandDetector(detectionCon=0.8,trackCon=0.8)
################################
wCam, hCam = 640, 480
################################

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed to open")
    exit()

cap.set(3, 640)
cap.set(4, 480)




detector: htm.HandDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

pTime = 0
pTime = 0

prevCount = 0

while True:
 
     success, img = cap.read()
     if not success or img is None:
        continue

     img = detector.findHands(img)
     lmList = detector.findPosition(img, draw=False)

     if len(lmList) !=0:
        
         fingers = detector.fingersUp()
         totalFingers = fingers.count(1)
        
     else:
        totalFingers = 0

        # Show finger image if available
     if totalFingers < len(overlayList):
            overlayImg = overlayList[totalFingers]
            h, w, _ = overlayImg.shape
            img[20:20+h, 20:20+w] = overlayImg

        # Draw number box
            cv2.rectangle(img, (20, 350), (170, 480), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(totalFingers), (45, 450),
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 8)

    # FPS Calculation
            cTime = time.time()
            fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
            pTime = cTime

            cv2.putText(img, f'FPS: {int(fps)}', (400, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            cv2.imshow("Finger Counter", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()


