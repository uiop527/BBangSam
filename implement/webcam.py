import cv2

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

if capture.isOpened():
    rval, frame1 = capture.read()
else:
    rval = False

i=0

while rval:
    cv2.imshow("VideoFrame", frame1)
    rval, frame = capture.read()
    
    
    frame1 = cv2.flip(frame,1) #좌우 반전
    frame2 = cv2.flip(frame,1)
    key = cv2.waitKey(20)
    if key == 27:       #esc
        break
    else:   #guideline 그리기
        cv2.ellipse(frame1, (320,320), (200,240), 0, 0, 360, (255,255,255), 2)
        cv2.putText(frame1, 'Place your face inside the line', (80,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
    
        if(key==32):                #spacebar
            cv2.imwrite('Picture' + str(i) + '.jpg', frame2)
            i += 1
            break;
            
        
capture.release()
cv2.destroyAllWindows()