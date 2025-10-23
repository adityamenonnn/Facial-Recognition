
import cv2 as c

capture=c.VideoCapture(0)

while True:
    ret,frame=capture.read()
    if ret==False:
        continue

    '''
    ret= capture.read()
    if ret==False:
        continue
    '''

    c.imshow("VideoFrame",frame)

    #Basically takes the input from user
    key=c.waitKey(1) & 0xFF

    #If key pressed is q, close the camera
    if key==ord("q"): 
        break
capture.release()
c.destroyAllWindows()
