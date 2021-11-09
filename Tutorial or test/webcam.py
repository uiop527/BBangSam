import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print('width: %d, height: %d' %(cap.get(3), cap.get(4)))

while(True):
    ret,frame = cap.read()
    
    if(ret):
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()