import cv2 as c
import numpy as np
import os  # <-- added

capture= c.VideoCapture(0) #Opens webcam

#Haarcascade--> Will show the different facial components of the face--> Eyes, Nose, Mouth and get measurements of it
face_cascade=c.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0
face_data=[] #All Face data is stored in this list
dataset_path="./face_dataset/"  # <-- changed from "/.face_dataset/"

file_name=input("Enter name") #will be the file name inside the face_dataset folder

# ensure folder exists  <-- added
os.makedirs(dataset_path, exist_ok=True)

while True:
    ret,frame=capture.read()
    if ret==False or frame is None:
        # show a blank window so you can still press 'q'
        c.imshow("faces", np.zeros((240,320,3), dtype=np.uint8))
        key_pressed=c.waitKey(1) & 0xFF
        if key_pressed==ord("q"):
            break
        continue

    #converting RGB image to Black and white--> easier to process the image for haarcascade
    grayscale=c.cvtColor(frame,c.COLOR_BGR2GRAY)

    # face_cascade_detectMultiScale returns coordinates of the face captured
    # returns the coordinates as (x,y, width, height)
    faces=face_cascade.detectMultiScale(grayscale,1.3,5) #(framename,scaling factor,no. of neighbours)

    if len(faces)==0:
        # show live feed even if no face found so window appears and 'q' works
        c.imshow("faces",frame)
        key_pressed=c.waitKey(1) & 0xFF
        if key_pressed==ord("q"):
            break
        continue

    k=1

    #Sorts faces list into descending order by area
    faces=sorted(faces, key=lambda x: x[2]*x[3], reverse= True)

    skip+=1

    for face in faces[:1]:
        x,y,width,height=face

        #padding of 5 spaces across the face to make a frame to show a clear user interface
        offset=5
        #subtracting padding and showing particular section of the face
        face_offset= frame[y-offset:y+height+offset,x-offset:x+width+offset]
        face_selection=c.resize(face_offset,(100,100)) #100x100 image is being created

        #when we get 10 different angles/ when we get 10 seconds of data of the face we add the data into face_data
        if skip%10==0:
            face_data.append(face_selection)
            print(len(face_data))

        c.imshow(str(k),face_selection)
        k+=1

        #top left{x,y} coordinates and bottom right{x+width,y+height} coordinates of the frame
        #RGB=255,0,0 to have a red color box
        c.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2) 
    
    c.imshow("faces",frame)

    key_pressed=c.waitKey(1) & 0xFF
    if key_pressed==ord("q"):
        break

# ===== Guard to avoid reshape error when no samples captured =====
if len(face_data) == 0:
    print("No face data captured. Keep the window open longer and ensure your face is detected.")
else:
    #converting list into numpy array for more computational functions
    face_data=np.array(face_data)
    #reshaping
    face_data=face_data.reshape((face_data.shape[0],-1))
    print(face_data.shape)

    np.save(dataset_path+file_name,face_data)
    print("File stored as {}".format(dataset_path+file_name+".npy"))

capture.release()
c.destroyAllWindows()