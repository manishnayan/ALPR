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
    os.makedirs(path_alpr_evidence)  # Create directory if it doesn't exist

def isduplicate(num1, num2):
    """
    Check if two license plate numbers are duplicates based on specific conditions.
    
    Args:
        num1 (str): The first license plate number.
        num2 (str): The second license plate number.
    
    Returns:
        bool: True if numbers are considered duplicates, otherwise False.
    """
    if len(num2) < 6:  # If second number is less than 6 characters, consider it a duplicate
        return True
    if num1[-4:] == num2[-4:]:  # Check if last four characters are the same
        return True
    if len(set(num1).intersection(set(num2))) > 5 and len(num1) != len(num2):  # Compare common characters
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
    layer_names = net.getLayerNames()  # Get names of all layers in the network
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]  # Get output layers
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
    global frame  # Global frame variable for further processing
    global lst  # Global list to hold detected characters
    global line1  # Global list for characters in the first line
    global line2  # Global list for characters in the second line

    label = str(classes[class_id])  # Get the label of the detected class
    tup = (x, y, label)  # Create a tuple of position and label

    # Separate detected characters into two lines based on y-coordinate
    if tup[1] < 60:
        line1.insert(0, tup)  # Insert into line1 if y-coordinate is less than 60
    else:
        line2.insert(0, tup)  # Otherwise, insert into line2

    # Draw rectangle and put label on the image
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)  # Draw bounding box
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)  # Draw label

def ysort(elem):
    """
    Sort elements by Y-coordinate.
    
    Args:
        elem: The element to sort.
    
    Returns:
        int: Y-coordinate value.
    """
    return elem[1]  # Return Y-coordinate for sorting

def xsort(elemt):
    """
    Sort elements by X-coordinate.
    
    Args:
        elemt: The element to sort.
    
    Returns:
        int: X-coordinate value.
    """
    return elemt[0]  # Return X-coordinate for sorting

def Reverse(lst):
    """
    Reverse a list.
    
    Args:
        lst (list): The list to reverse.
    
    Returns:
        list: Reversed list.
    """
    lst.reverse()  # Reverse the list in place
    return lst  # Return reversed list

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
    global frame2  # Global frame variable for detected plates
    global plate_number  # Global variable to hold the detected plate number

    try:
        # Sort detected characters by Y-coordinate and then by X-coordinate
        lst.sort(key=ysort)  # Sort lst based on Y-coordinate
        line1.sort(key=xsort)  # Sort line1 based on X-coordinate
        line2.sort(key=xsort)  # Sort line2 based on X-coordinate

        # Combine detected characters from two lines into a single list
        for t1 in range(len(line1)):
            lst.insert(0, line1[t1][2])  # Insert characters from line1 into lst

        for t2 in range(len(line2)):
            lst.insert(0, line2[t2][2])  # Insert characters from line2 into lst

        # Join characters to form the license plate number
        str1 = ''.join(Reverse(lst))  # Reverse the list and join characters
        plate_number = str1  # Update the global plate_number

    except:
        exception = 'exception occurs-'  # Handle exceptions gracefully

# Load the classes from the OCR file
with open("ocr.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]  # Read class names from file

# Generate random colors for each class for visualization
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))  # Random colors for bounding boxes

# Load the neural network models for OCR and plate detection
net = cv2.dnn.readNet("ocr.weights", "ocr.cfg")  # Load OCR model
net1 = cv2.dnn.readNet("plates.weights", "plates.cfg")  # Load plate detection model

class VideoCapture:
    """
    Class for capturing video frames.
    """
    def __init__(self):
        # Initialize video capture with a video file
        self.name = "test2.mp4"  # Video file name
        self.cap = cv2.VideoCapture(self.name)  # Create video capture object
        self.q = queue.Queue()  # Initialize a queue to hold frames
        t = threading.Thread(target=self.reader)  # Create a thread to read frames
        t.daemon = True  # Daemon thread will exit when the program does
        t.start()  # Start the thread

    def reader(self):
        """
        Read frames from the video and put them in a queue.
        """
        while True:
            ret, frame = self.cap.read()  # Read a frame from the video
            if not ret:
                # If end of video is reached, restart the video after a short delay
                self.cap = cv2.VideoCapture(self.name)  # Restart video capture
                time.sleep(2)  # Delay before restarting
                continue
            if not self.q.empty():  # If the queue is not empty
                try:
                    self.q.get_nowait()  # Discard the oldest frame
                except queue.Empty:
                    pass  # Ignore if the queue is empty
            self.q.put(frame)  # Add the new frame to the queue

    def read(self):
        """
        Get a frame from the queue.
        
        Returns:
            Frame from the video.
        """
        return self.q.get()  # Retrieve a frame from the queue

# Initialize global variables for processing
lst = []  # List to hold detected characters
line1 = []  # List for characters in the first line
line2 = []  # List for characters in the second line
previous_number = None  # Variable to hold the previous plate number
_time = str(int(time.time()))  # Current time as a string
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video output
videoWriter = cv2.VideoWriter("output/" + _time + '.mp4', fourcc, 12, (1920, 1080))  # Initialize video writer
plate_number = " "  # Placeholder for detected plate number
framecount = 0  # Frame counter for processing
count = 0  # Global counter for frame processing
print("processing...")  # Indicate processing start

# Create a DataFrame to store results
df = pd.DataFrame(columns=['Sr_No', 'Date', 'Time', 'License_plate_number', 'Cropped_license_plate_image_name'])  # DataFrame for results
sr_no = 0  # Serial number for entries

# Initialize video capture object
cap = VideoCapture()  # Create an instance of VideoCapture

# Main loop for processing video frames
while True:
    try:
        frame2 = cap.read()  # Get the next frame from the video
        count += 1  # Increment the frame count
        if count % 3 == 0:  # Process every third frame
            s = time.time()  # Start timing
            plate = None  # Placeholder for detected plate
            if frame2 is None:  # Check if the frame is valid
                print("video finished")  # Indicate end of video
                break  # Exit loop
            frame2 = frame2[0:1080, 0:1920]  # Crop frame to desired resolution

            Width1 = frame2.shape[1]  # Get frame width
            Height1 = frame2.shape[0]  # Get frame height

            # Preprocess the frame for plate detection
            blob1 = cv2.dnn.blobFromImage(frame2, 0.003, (416, 416), (0, 0, 0), True, crop=False)  # Prepare input for the network
            net1.setInput(blob1)  # Set the input to the network
            outs1 = net1.forward(get_output_layers(net1))  # Perform forward pass

            class_ids1 = []  # List for class IDs
            confidences1 = []  # List for confidence scores
            boxes1 = []  # List for bounding boxes
            conf_threshold1 = 0.4  # Confidence threshold
            nms_threshold1 = 0.5  # Non-max suppression threshold

            # Parse the output from the plate detection network
            for out in outs1:
                for detection in out:
                    scores = detection[5:]  # Get class scores
                    class_id = np.argmax(scores)  # Get the class ID with the highest score
                    confidence = scores[class_id]  # Get the confidence for that class
                    if confidence > 0.4:  # Check if confidence is above threshold
                        center_x = int(detection[0] * Width1)  # Calculate center x-coordinate
                        center_y = int(detection[1] * Height1)  # Calculate center y-coordinate
                        w = int(detection[2] * Width1)  # Calculate width
                        h = int(detection[3] * Height1)  # Calculate height
                        x = center_x - w / 2  # Calculate top-left x-coordinate
                        y = center_y - h / 2  # Calculate top-left y-coordinate
                        class_ids1.append(class_id)  # Add class ID to list
                        confidences1.append(float(confidence))  # Add confidence to list
                        boxes1.append([x, y, w, h])  # Add bounding box to list

            # Apply non-maximum suppression to remove overlapping bounding boxes
            indices1 = cv2.dnn.NMSBoxes(boxes1, confidences1, conf_threshold1, nms_threshold1)  # Perform NMS
            for i in indices1:  # Iterate through remaining indices
                box = boxes1[i]  # Get bounding box
                print("----------", box)  # Print box for debugging
                x = box[0]  # Get x-coordinate
                y = box[1]  # Get y-coordinate
                w = box[2]  # Get width
                h = box[3]  # Get height
                draw_prediction1(frame2, class_ids1[i], confidences1[i], round(x), round(y), round(w), round(h))  # Draw prediction
                x, y, w, h = int(x), int(y), int(w), int(h)  # Round coordinates

            # If no plates are detected, continue to the next frame
            if plate_number == "" or len(plate_number) < 6 or plate_number.isalnum() == False:
                previous_number = plate_number  # Update previous plate number
                cv2.putText(frame2, plate_number, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)  # Display plate number
                framecount = 0  # Reset frame count
                cv2.waitKey(1)  # Wait for a short period
                continue  # Skip to the next frame

            # Check if the detected plate number is a duplicate
            if previous_number and isduplicate(previous_number, plate_number):
                previous_number = plate_number  # Update previous plate number
                cv2.putText(frame2, plate_number, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)  # Display plate number
                framecount = 0  # Reset frame count
                cv2.waitKey(1)  # Wait for a short period
                continue  # Skip to the next frame

            # If a valid plate number is detected, save the plate image and details
            if plate_number != "":
                plate = frame2[y:y + h, x:x + w]  # Crop the plate from the frame
                print("detected")  # Indicate detection
                if framecount > 20:  # Check if frame count exceeds threshold
                    _time = str(int(time.time()))  # Get current time for filename
                    framecount = 0  # Reset frame count
                    sr_no += 1  # Increment serial number
                    data = [sr_no, strftime("%Y-%m-%d", gmtime()), strftime("%H:%M:%S", gmtime()), plate_number, plate_number + "_" + _time + ".jpg"]  # Prepare data
                    df.loc[len(df)] = data  # Add data to DataFrame
                    df.to_csv('output/alpr_result.csv')  # Save results to CSV
                    try:
                        cv2.imwrite(path_alpr_evidence + "/" + plate_number + "_" + _time + ".jpg", plate)  # Save cropped plate image
                    except Exception as e:
                        print(e)  # Print any exception encountered

                    cv2.putText(frame2, plate_number, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)  # Display plate number
                    framecount = 0  # Reset frame count
                else:
                    framecount += 1  # Increment frame count

            previous_number = plate_number  # Update previous plate number
            print(plate_number)  # Print detected plate number
            videoWriter.write(frame2)  # Write frame to video output
            cv2.putText(frame2, plate_number, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)  # Display plate number
            cv2.imshow("ALPR", frame2)  # Show frame in window
            cv2.waitKey(1)  # Wait for a short period
            line1 = []  # Clear line1 for next frame
            line2 = []  # Clear line2 for next frame
            lst = []  # Clear lst for next frame

    except Exception as e:
        print(e)  # Print any exception encountered

# Release resources
cap.release()  # Release video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
videoWriter.release()  # Release video writer
