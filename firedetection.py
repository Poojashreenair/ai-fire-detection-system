import cv2
import numpy as np
import time
import pyttsx3
import sendmail
from pygame import mixer
import threading
engine = pyttsx3.init()
engine.setProperty("rate", 110)
mixer.init()

def buzzer():
    sound = mixer.Sound('alarm.wav')
    sound.play()



def voicebuzzer():
    engine.say('Fire detected')
    engine.runAndWait()
    
    

net = cv2.dnn.readNet("fire.weights","fire.cfg") #Tiny Yolo
classes = []
with open("fire.names","r") as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


#loading image
cap=cv2.VideoCapture(0) #0 for 1st webcam
font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0
count_id = 0
obj_id = 0
while True:
    _,frame= cap.read() # 
    frame_id+=1
    
    height,width,channels = frame.shape
    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(224,224),(0,0,0),True,crop=False) #reduce 416 to 320    

        
    net.setInput(blob)
    outs = net.forward(outputlayers)
    #print(outs[1])


    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]
    TrackedIDs = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                count_id += 1
                #object detected
                
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #get ID
                Id = int(obj_id)                
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                
                #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])

            confidence= confidences[i]
            color = (0,0,255)
            cv2.circle(frame,(center_x,center_y),1,(0,255,0),2)            
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
            if(label == 'fire' and count_id % 15 == 0):
                cv2.imwrite("final.jpg",frame)
                img_path = "final.jpg"
                
                threading.Thread(target=buzzer).start()
                threading.Thread(target=voicebuzzer).start()
                try:
                    sendmail.sendalert()
                except:
                    print("could not send mail")
                alert = 0
    
    cv2.putText(frame,"Monitoring",(10,50),font,2,(0,0,255),2)    
    cv2.imshow("FIRE DETECTION",frame)
    key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
    
    if key == 27: #esc key stops the process
        break
    
cap.release()    
cv2.destroyAllWindows()

