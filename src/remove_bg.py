import numpy as np, cv2

cap = cv2.VideoCapture('/home/rica/Documents/ret/ret.h264')
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000/fps)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
output = cv2.VideoWriter("/home/rica/Documents/ret/ret_nobg.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480)) 
  
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    fgmask = fgbg.apply(frame)
    output.write(fgmask)
    cv2.imshow('bgsub',fgmask)
    
    if cv2.waitKey(1) & 0xff == 27:
        break
cap.release()
output.release()
cv2.destroyAllWindows()