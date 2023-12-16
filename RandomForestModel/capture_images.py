import os
import cv2


DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


number_of_classes = 26
dataset_size = 100

def computer_vision_helper(curr_frame, number_of_signs, class_num):
    # Helper function that collects the image and appropriately name it
    cv2.imshow('frame', curr_frame)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(DATA_DIR, str(class_num), '{}.jpg'.format(number_of_signs)), curr_frame)


# Initialize the laptop camera
cap = cv2.VideoCapture(0)


# For loop that loops the the number of desired signs and collects images
for class_num in range(number_of_classes):
    # Check if "data" direcotry exists
    if not os.path.exists(os.path.join(DATA_DIR, str(class_num))):
        os.makedirs(os.path.join(DATA_DIR, str(class_num)))

    # The terminal will print out what class number we are currently collecting
    print('Collecting data for class {}'.format(class_num))

    # done = False
    while True:
        # once camera view has popped up, the user will press enter to begin collecting signs
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Enter" or "Return" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 6,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('\r'):
            break

    image_num = 0
    while image_num < dataset_size:
        ret, frame = cap.read()
        
        # collect the image and process it into the right file
        computer_vision_helper(frame, image_num, class_num)
        
        image_num += 1

cap.release()
cv2.destroyAllWindows()


