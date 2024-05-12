from pprint import pprint
import cv2
import time

# Configure trackers
trackers = [cv2.legacy.TrackerBoosting_create,
            cv2.legacy.TrackerMIL_create,
            cv2.legacy.TrackerKCF_create,
            cv2.legacy.TrackerTLD_create,
            cv2.legacy.TrackerMedianFlow_create,
            #cv2.legacy.TrackerGOTURN_create, #goturn.caffemodel, goturnportxt are required
            cv2.legacy.TrackerCSRT_create,
            cv2.legacy.TrackerMOSSE_create]
trackerIdx = 0
tracker = None
tracker2 = None
isFirst = True
delay=1
win_name = 'Tracking APIs'

src = 0
video = cv2.VideoCapture(src)


print('Processing frames')
while video.isOpened():
    ret, frame = video.read()
    img_draw = frame.copy()
    if tracker is None: # ROI is unset initial step
        cv2.putText(img_draw, "Press the Space to set ROI!!", \
            (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
    else: # in case of tracker set
        ok, bbox = tracker.update(frame)   # Update bbox
        (x,y,w,h) = bbox

        if tracker2 != None:
            ok2, bbox2 = tracker2.update(frame)
            (x2,y2,w2,h2) = bbox2
            if ok2:
                cv2.rectangle(img_draw, (int(x2), int(y2)), (int(x2 + w2), int(y2 + h2)), \
                        (0,255,0), 2, 1)
        if ok: # Draw a newly updated bounding box with rect
            cv2.rectangle(img_draw, (int(x), int(y)), (int(x + w), int(y + h)), \
                          (0,255,0), 2, 1)
            print('tracking....')
        else : # On a tracking fail
            cv2.putText(img_draw, "Tracking fail.", (100,80), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv2.LINE_AA)
            tracker = None
            bbox = None
            isFirst = True
    trackerName = tracker.__class__.__name__
    cv2.putText(img_draw, str(trackerIdx) + ":"+trackerName , (100,20), \
                 cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0),2,cv2.LINE_AA)

    cv2.imshow(win_name, img_draw)
    key = cv2.waitKey(delay) & 0xff
    # Wait key input and configure tracker
    if key == ord(' '): #Select ROI and set default tracker
        isFirst = False
        roi = cv2.selectROI(win_name, frame, False)  # Drag an area of ROI
        print('Select ROI')
        if roi[2] and roi[3]:         # when roi is set (area), initialize tracker with the ROI
            tracker = trackers[trackerIdx]()
            isInit = tracker.init(frame, roi)
            print('tracker is initalized')
    elif key == ord('a'): #Add another tracker
        roi2 = cv2.selectROI(win_name, frame, False)
        print('Select another ROI')
        if roi2[2] and roi2[3]:
            tracker2 = trackers[trackerIdx]()
            tracker2.init(frame, roi2)
    elif key in range(48, 55): #key (0~6) change tracker and init with the existing bounding box
        trackerIdx = key-48 
        if bbox is not None:
            tracker = trackers[trackerIdx]()
            isInit = tracker.init(frame, bbox)
    elif key == 27 : #ESC for exit 
        break

video.release()
cv2.destroyAllWindows()
print('Terminated')
