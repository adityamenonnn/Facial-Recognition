#RUN IN TERMINAL AS
#cd /Users/adityamenon/Downloads/Facial
#python3 FaceRecognition.py


import numpy as np
import cv2 as c
import os
capture= c.VideoCapture(0) #Opens webcam

#KNN Code-->https://www.youtube.com/watch?v=4rAlxRGBNz0&list=PLS1QulWo1RIbp_ImnSEWEMRLnJVfEc-GR&index=5

# Distance function
def distance(v1, v2):
    # Euclidean
    return np.sqrt(((v1 - v2) ** 2).sum())

# KNN function
def knn(train, test, k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label
        ix = train[i, :-1]
        iy = train[i, -1]

        # Compute the distance from test point
        d = distance(test, ix)
        dist.append([d, iy])

    # Sort based on distance and get top k
    dk = sorted(dist, key=lambda x: x[0])[:k]

    # Retrieve only the labels
    labels = np.array(dk)[:, -1]

    # Get frequencies of each label
    output = np.unique(labels, return_counts=True)

    # Find max frequency and corresponding label
    index = np.argmax(output[1])
    return output[0][index]


# use the existing capture above (remove duplicate)
face_cascade=c.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path="./face_dataset/" #the file in the directory where all faces are stored

face_data=[]
labels=[] 
class_id=0 #label for the file
names={} #id and name mapping

#Data Preperation

for file in os.listdir(dataset_path):
    if file.endswith(".npy"):
        names[class_id]=file[:-4] #the key=id number, the value is the file name without the .npy which is the name
        data_item=np.load(dataset_path+file)
        face_data.append(data_item)

        target=class_id* np.ones((data_item.shape[0],)) # class id mulitploed by an array of ones with the length of the data item
        class_id+=1
        labels.append(target)

face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))  # fixed: label -> labels
print(face_dataset)
print(face_labels)

trainset= np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

font=c.FONT_HERSHEY_SIMPLEX

while True:
    ret,frame=capture.read()        # fixed: proper unpack
    if ret==False:                  # avoid using an empty frame
        continue

    #converting RGB image to Black and white--> easier to process the image for haarcascade
    grayscale=c.cvtColor(frame,c.COLOR_BGR2GRAY)

    #faces is a list 
    # face_cascade_detectMultiScale returns coordinates of the face captured
    # returns the coordinates as (x,y, width, height) where x,y is the coordinates of the top left corner of the square
    faces=face_cascade.detectMultiScale(grayscale,1.3,5) #(framename,scaling factor,no. of neighbours)


    for face in faces[:1]:
        x,y,width,height=face

        #padding of 5 spaces across the face to make a frame to show a clear user interface
        offset=5
        #subtracting padding and showing particular section of the face
        face_offset= frame[y-offset:y+height+offset,x-offset:x+width+offset]
        face_section=c.resize(face_offset,(100,100)) #100x100 image is being created

        out=knn(trainset,face_section.flatten()) #array of one column

        #top left{x,y} coordinates and bottom right{x+width,y+height} coordinates of the frame
        #BGR=0,255,0 to have a red color box
        c.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2) 
        c.putText(frame, names[int(out)],(x,y-10),c.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,c.LINE_AA)
    
    c.imshow("faces",frame)

    key_pressed=c.waitKey(1) & 0xFF
    if key_pressed==ord("q"):
        break

capture.release()
c.destroyAllWindows()
