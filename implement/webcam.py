import cv2

capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if capture.isOpened():
    rval, frame = capture.read()
else:
    rval = False

i=0
    
while rval:
    cv2.imshow("VideoFrame", frame)
    rval, frame = capture.read()
    frame = cv2.flip(frame,1) #좌우 반전
    key = cv2.waitKey(20)
    if key == 27:       #esc
        break
    else:   #guideline 그리기
        #왼쪽 아래 line
       
        cv2.line(img=frame, pt1=(140, 480), pt2=(140, 450), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(140, 430), pt2=(140, 400), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(140, 380), pt2=(140, 350), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(140, 335), pt2=(160, 313), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(170, 310), pt2=(200, 300), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(210, 300), pt2=(240, 300), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(240, 300), pt2=(220, 290), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(210, 280), pt2=(205, 270), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(203, 265), pt2=(200, 252), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(200, 245), pt2=(200, 230), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(200, 220), pt2=(200, 205), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(200, 195), pt2=(205, 180), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(205, 170), pt2=(210, 160), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(210, 150), pt2=(215, 145), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(220, 140), pt2=(230, 135), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(235, 132), pt2=(242, 130), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(250, 130), pt2=(260, 128), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(270, 128), pt2=(280, 128), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(290, 128), pt2=(310, 128), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
      
        #오른쪽 아래 line 
        
        cv2.line(img=frame, pt1=(500, 480), pt2=(500, 450), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(500, 430), pt2=(500, 400), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(500, 380), pt2=(500, 350), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(500, 335), pt2=(480, 313), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(470, 310), pt2=(440, 300), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(430, 300), pt2=(400, 300), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(400, 300), pt2=(420, 290), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(430, 280), pt2=(435, 270), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(437, 265), pt2=(440, 252), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(440, 245), pt2=(440, 230), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(440, 220), pt2=(440, 205), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(440, 195), pt2=(435, 180), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(435, 170), pt2=(430, 160), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(430, 150), pt2=(425, 145), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(420, 140), pt2=(410, 135), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(405, 132), pt2=(398, 130), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(390, 130), pt2=(380, 128), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(370, 128), pt2=(350, 128), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        cv2.line(img=frame, pt1=(340, 128), pt2=(320, 128), color=(220, 255, 0), thickness=2, lineType=8, shift=0)
        
        if(key==32):                #spacebar
            cv2.imwrite('Picture' + str(i) + '.jpg', frame)
            i += 1
            break;
            
        
capture.release()
cv2.destroyAllWindows()