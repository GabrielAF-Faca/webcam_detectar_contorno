import cv2
from time import time

cascPath = './cascade.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0)

previous = time()
delta = 0

can_foto = True

while True:
    _, frame = cap.read()
    # convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # create a binary thresholded image
    _, binary = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY)
    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # draw all contours
    image = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    
    if len(faces) and can_foto:

        current = time()
        delta += current - previous
        previous = current
        
        print(delta)

        if delta > 10:
            can_foto = False
            # Operations on image
            # Reset the time counter
            print("[INFO] Object found. Saving locally.") 
            cv2.imwrite('foto.jpg', binary) 
            
            delta = 0
    else:
        previous = time()
        delta = 0
    
    # show the images
    cv2.imshow("image", image)
    cv2.imshow("binary", binary)
    
    
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()