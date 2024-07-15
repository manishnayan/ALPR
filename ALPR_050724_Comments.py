import cv2
import numpy as np
import os
import time
import pandas as pd
import queue
import threading
from time import gmtime, strftime

# Initialize a global count variable for frame processing
count = 1

# Define the root path directory (user's home directory)
root_path_directory = os.path.expanduser('~')

# Define the directory for storing cropped plate images
path_alpr_evidence = "cropped_plate_image"
if not os.path.exists(path_alpr_evidence):
    os.makedirs(path_alpr_evidence)

def isduplicate(num1, num2):
    """
    Check if two license plate numbers are duplicates based on specific conditions.
    
    Args:
        num1 (str): The first license plate number.
        num2 (str): The second license plate number.
    
    Returns:
        bool: True if numbers are considered duplicates, otherwise False.
    """
    if len(num2) < 6:
        return True
    if num1[-4:] == num2[-4:]:
        return True
    if len(set(num1).intersection(set(num2))) > 5 and len(num1) != len(num2):
        return True
    else:
        return False

def get_output_layers(net):
    """
    Get the output layers of the network.
    
    Args:
        net: The neural network model.
    
    Returns:
        list: List of output layer names.
    """
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, w, h):
    """
    Draw bounding box and label on the image for detected characters.
    
    Args:
        img: The image to draw on.
        class_id: Class ID of the detected object.
        confidence: Confidence score of the detection.
        x: X-coordinate of the top-left corner of the bounding box.
        y: Y-coordinate of the top-left corner of the bounding box.
        w: Width of the bounding box.
        h: Height of the bounding box.
    """
    global frame
    global lst
    global line1
    global line2

    label = str(classes[class_id])
    tup = (x, y, label)

    # Separate detected characters into two lines based on y-coordinate
    if tup[1] < 60:
        line1.insert(0, tup)
    else:
        line2.insert(0, tup)

    # Draw rectangle and put label on the image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

def ysort(elem):
    """
    Sort elements by Y-coordinate.
    
    Args:
        elem: The element to sort.
    
    Returns:
        int: Y-coordinate value.
    """
    return elem[1]

def xsort(elemt):
    """
    Sort elements by X-coordinate.
    
    Args:
        elemt: The element to sort.
    
    Returns:
        int: X-coordinate value.
    """
    return elemt[0]

def Reverse(lst):
    """
    Reverse a list.
    
    Args:
        lst (list): The list to reverse.
    
    Returns:
        list: Reversed list.
    """
    lst.reverse()
    return lst

def draw_prediction1(img, class_id, confidence, x, y, w, h):
    """
    Draw bounding box and label on the image for detected license plates
    and update the plate number.
    
    Args:
        img: The image to draw on.
        class_id: Class ID of the detected object.
        confidence: Confidence score of the detection.
        x: X-coordinate of the top-left corner of the bounding box.
        y: Y-coordinate of the top-left corner of the bounding box.
        w: Width of the bounding box.
        h: Height of the bounding box.
    """
    global frame2
    global plate_number

    try:
        # Sort detected characters by Y-coordinate and then by X-coordinate
        lst.sort(key=ysort)
        line1.sort(key=xsort)
        line2.sort(key=xsort)

        # Combine detected characters from two lines into a single list
        for t1 in range(len(line1)):
            lst.insert(0, line1[t1][2])

        for t2 in range(len(line2)):
            lst.insert(0, line2[t2][2])

        # Join characters to form the license plate number
        str1 = ''.join(Reverse(lst))
        plate_number = str1

    except:
        exception = 'exception occurs-'

# Load the classes from the OCR file
with open("ocr.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate random colors for each class for visualization
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Load the neural network models for OCR and plate detection
net = cv2.dnn.readNet("ocr.weights", "ocr.cfg")
net1 = cv2.dnn.readNet("plates.weights", "plates.cfg")

class VideoCapture:
    """
    Class for capturing video frames.
    """
    def __init__(self):
        # Initialize video capture with a video file
        self.name = "test2.mp4"
        self.cap = cv2.VideoCapture(self.name)
        self.q = queue.Queue()
        t = threading.Thread(target=self.reader)
        t.daemon = True
        t.start()

    def reader(self):
        """
        Read frames from the video and put them in a queue.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                # If end of video is reached, restart the video after a short delay
                self.cap = cv2.VideoCapture(self.name)
                time.sleep(2)
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        """
        Get a frame from the queue.
        
        Returns:
            Frame from the video.
        """
        return self.q.get()

# Initialize global variables for processing
lst = []
line1 = []
line2 = []
previous_number = None
_time = str(int(time.time()))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter("output/" + _time + '.mp4', fourcc, 12, (1920, 1080))
plate_number = " "
framecount = 0
count = 0
print("processing...")

# Create a DataFrame to store results
df = pd.DataFrame(columns=['Sr_No', 'Date', 'Time', 'License_plate_number', 'Cropped_license_plate_image_name'])
sr_no = 0

# Initialize video capture object
cap = VideoCapture()

# Main loop for processing video frames
while True:
    try:
        frame2 = cap.read()
        count += 1
        if count % 3 == 0:
            s = time.time()
            plate = None
            if frame2 is None:
                print("video finished")
                break
            frame2 = frame2[0:1080, 0:1920]

            Width1 = frame2.shape[1]
            Height1 = frame2.shape[0]

            # Preprocess the frame for plate detection
            blob1 = cv2.dnn.blobFromImage(frame2, 0.003, (416, 416), (0, 0, 0), True, crop=False)
            net1.setInput(blob1)
            outs1 = net1.forward(get_output_layers(net1))

            class_ids1 = []
            confidences1 = []
            boxes1 = []
            conf_threshold1 = 0.4
            nms_threshold1 = 0.5

            # Parse the output from the plate detection network
            for out in outs1:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.4:
                        center_x = int(detection[0] * Width1)
                        center_y = int(detection[1] * Height1)
                        w = int(detection[2] * Width1)
                        h = int(detection[3] * Height1)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids1.append(class_id)
                        confidences1.append(float(confidence))
                        boxes1.append([x, y, w, h])

            # Apply non-maximum suppression to remove overlapping bounding boxes
            indices1 = cv2.dnn.NMSBoxes(boxes1, confidences1, conf_threshold1, nms_threshold1)
            for i in indices1:
                box = boxes1[i]
                print("----------", box)
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                draw_prediction1(frame2, class_ids1[i], confidences1[i], round(x), round(y), round(w), round(h))
                x, y, w, h = int(x), int(y), int(w), int(h)

            # If no plates are detected, continue to the next frame
            if plate_number is None:
                continue

            # If the detected plate number is not a duplicate, process the frame
            if not isduplicate(previous_number, plate_number):
                previous_number = plate_number
                cv2.putText(frame2, plate_number, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)
                framecount = 0
                cv2.waitKey(1)
                continue

            if plate_number != "":
                plate = frame2[y:y + h, x:x + w]
                print("detected")
                if framecount > 20:
                    _time = str(int(time.time()))
                    framecount = 0
                    sr_no += 1
                    data = [sr_no, strftime("%Y-%m-%d", gmtime()), strftime("%H:%M:%S", gmtime()), plate_number, plate_number + "_" + _time + ".jpg"]
                    df.loc[len(df)] = data
                    df.to_csv('output/alpr_result.csv')
                    try:
                        cv2.imwrite(path_alpr_evidence + "/" + plate_number + "_" + _time + ".jpg", plate)
                    except Exception as e:
                        print(e)

                    cv2.putText(frame2, plate_number, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)
                    framecount = 0
                else:
                    framecount += 1

            previous_number = plate_number
            print(plate_number)
            videoWriter.write(frame2)
            cv2.putText(frame2, plate_number, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)
            cv2.imshow("ALPR", frame2)
            cv2.waitKey(1)
            line1 = []
            line2 = []
            lst = []

    except Exception as e:
        print(e)

# Release resources after processing
cap.release()
cv2.destroyAllWindows()
videoWriter.release()

