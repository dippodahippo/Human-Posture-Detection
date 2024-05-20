import threading
from customtkinter import *
from pyglet import font
from PIL import Image

def slouch():
    import cv2
    import mediapipe as mp
    import pyduinointegr.pyduino_connection as pyd
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    
    pyd.show_ports()
    port = pyd.select_port(6)
    pyd.open_port(port)
    
    cap = cv2.VideoCapture(0)

    reference_line = 0 
    reference_set = False
    timer = 0

    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            nose_landmark = landmarks.landmark[4]
            nose_x, nose_y = int(nose_landmark.x * frame.shape[1]), int(nose_landmark.y * frame.shape[0])

            if not reference_set:
                chin_landmark = landmarks.landmark[15]  
                chin_x, chin_y = int(chin_landmark.x * frame.shape[1]), int(chin_landmark.y * frame.shape[0])
                reference_line = chin_y + 20
                timer += 1

                if timer == 5 * cap.get(cv2.CAP_PROP_FPS):
                    reference_set = True
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)
            cv2.line(frame, (0, reference_line), (frame.shape[1], reference_line), (0, 255, 255), 2)
            if reference_set and nose_y > reference_line:
                text = "Slouching"
                color = (0, 0, 255)
                pyd.send_data(0)
            else:
                text = "Straight"
                color = (0, 255, 0) 
                pyd.send_data(1)

            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Slouch Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    pyd.close_port()
    
def yogapose():
    import mediapipe as mp
    import cv2
    import numpy as np
    import tensorflow as tf
    import pyduinointegr.pyduino_connection as pyd

    pyd.show_ports()
    port = pyd.select_port(6)
    pyd.open_port(port)

    model = tf.keras.models.load_model("yoga_pose_model")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5)

    smoothing_factor = 0.2

    prev_prediction = np.zeros(21)

    def getpose(image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks is not None:
            keypoints = np.zeros((len(results.pose_landmarks.landmark), 2))
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[i] = [landmark.x, landmark.y]
            return keypoints, results.pose_landmarks
        else:
            return None, None

    def predpose(keypoints):
        keypoints = keypoints / 640 
        prediction = model.predict(np.expand_dims(keypoints, axis=0))[0]
        return prediction

    def all_keypoints_visible(keypoints):
        return keypoints is not None and len(keypoints) == 33

    def smooth_predictions(current_prediction):
        nonlocal prev_prediction
        smoothed_prediction = smoothing_factor * current_prediction + (1 - smoothing_factor) * prev_prediction
        prev_prediction = smoothed_prediction
        return smoothed_prediction
    
    def get_custom_label(index):
        if index == 0:
            return "Anjaneyasana"
        elif index == 1:
            return "Adho Mukha Svasana"
        elif index == 2:
            return "ardha chakrasana"
        elif index == 3:
            return "bhujangasana"
        elif index == 4:
            return "chakrasana"
        elif index == 5:
            return "Dhanurasana"
        elif index == 6:
            return "malasana"
        elif index == 7:
            return "Naukasana"
        elif index == 8:
            return "paschimottasana"
        elif index == 9:
            return "shavasana"
        elif index == 10:
            return "Setu Bandha Sarvagasana"
        elif index == 11:
            return "tadasana"
        elif index == 12:
            return "trikonasana"
        elif index == 13:
            return "uttanasana"
        elif index == 14:
            return "ustrasana"
        elif index == 15:
            return "Utkatasana"
        elif index == 16:
            return "vajrasana"
        elif index == 17:
            return "Virabhadrasan 1"
        elif index == 18:
            return "Virabhadrasan 2"
        elif index == 19:
            return "Virabhadrasan 3"
        elif index == 20:
            return "vrikshasana"
        else:
            return "Unknown Label"
    
    def detectpose(image):
        keypoints, landmarks = getpose(image)
        if keypoints is None:
            error_text = "Error: Person out of frame"
            cv2.putText(image, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            return image
        
        image_height, image_width, _ = image.shape
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        
        prediction = predpose(keypoints)
        smoothed_prediction = smooth_predictions(prediction)
        max_confidence = np.max(smoothed_prediction)
        
        if max_confidence > 0.95:
            pyd.send_data(1)
        else:
            pyd.send_data(0)
        selected_indices = [3,6,11,12,13,17,20]
        line_spacing = 30
        for i, confidence in enumerate(smoothed_prediction):
            if i in selected_indices:
                label = get_custom_label(i)
                accuracy = (confidence / max_confidence) * 100
                accuracy_text = f"{label}: {round(accuracy, 2)}%"
                y_coordinate = 30 + selected_indices.index(i) * line_spacing
                cv2.putText(image, accuracy_text, (10, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image

    def main():
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = detectpose(frame)
            cv2.imshow('Yoga Pose Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pyd.close_port()
        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

def gympose():
    import mediapipe as mp
    import cv2
    import numpy as np
    import tensorflow as tf
    import pyduinointegr.pyduino_connection as pyd

    pyd.show_ports()
    port = pyd.select_port(6)
    pyd.open_port(port)

    model = tf.keras.models.load_model("gym_pose_model")

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5)

    smoothing_factor = 0.2

    prev_prediction = np.zeros(6)

    def getpose(image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if results.pose_landmarks is not None:
            keypoints = np.zeros((len(results.pose_landmarks.landmark), 2))
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[i] = [landmark.x, landmark.y]
            return keypoints, results.pose_landmarks
        else:
            return None, None

    def predpose(keypoints):
        keypoints = keypoints / 640 
        prediction = model.predict(np.expand_dims(keypoints, axis=0))[0]
        return prediction

    def all_keypoints_visible(keypoints):
        return keypoints is not None and len(keypoints) == 33

    def smooth_predictions(current_prediction):
        nonlocal prev_prediction
        if prev_prediction.shape != current_prediction.shape:
            prev_prediction = np.zeros_like(current_prediction)
        smoothed_prediction = smoothing_factor * current_prediction + (1 - smoothing_factor) * prev_prediction
        prev_prediction = smoothed_prediction
        return smoothed_prediction
    
    def get_custom_label(index):
        if index == 0:
            return "Bench Up"
        elif index == 1:
            return "Shoulder Press Down"
        elif index == 2:
            return "Shoulder Press Up"
        elif index == 3:
            return "Bench Down"
        elif index == 4:
            return "Squat Down"
        elif index == 5:
            return "Squat Up"
        
    def detectpose(image):
        keypoints, landmarks = getpose(image)
        if keypoints is None:
            error_text = "Error: Person out of frame"
            cv2.putText(image, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            return image
        
        image_height, image_width, _ = image.shape
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        
        prediction = predpose(keypoints)
        smoothed_prediction = smooth_predictions(prediction)
        max_confidence = np.max(smoothed_prediction)
        
        if max_confidence > 0.95:
            pyd.send_data(1)
        else:
            pyd.send_data(0)
        selected_indices = [1,2,4,5]
        line_spacing = 30
        for i, confidence in enumerate(smoothed_prediction):
            if i in selected_indices:
                label = get_custom_label(i)
                accuracy = (confidence / max_confidence) * 100
                accuracy_text = f"{label}: {round(accuracy, 2)}%"
                y_coordinate = 30 + selected_indices.index(i) * line_spacing
                cv2.putText(image, accuracy_text, (10, y_coordinate), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image

    def main():
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = detectpose(frame)
            cv2.imshow('Gym Pose Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pyd.close_port()
        cap.release()
        cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()

# variables
HEIGHT = 800
WIDTH = 800
FRAME_WIDTH = 750
FRAME_HEIGHT = 100
TEXT_FRAME_HEIGHT = 300
LIGHT_BLUE = "#C6DEF2"
BLUE = "#92BEE3"
DARK_BLUE = "#6AABD2"
font.add_file("Fonts/Bebas_Neue/BebasNeue-Regular.ttf")

desc_text = """1. Master Yoga Poses: Get real-time feedback on your yoga form to ensure proper alignment and maximize your practice.
2. Maintain Perfect Posture: Our model monitors your posture throughout the day, helping you stay upright and avoid slouching.
3. Level Up Your Gym Workouts: See your gym exercises graded in real-time, ensuring you perform them safely and effectively with proper form."""

# setting up the window

set_appearance_mode("System")
set_default_color_theme("blue")

app = CTk(fg_color=BLUE)
app.geometry(f"{WIDTH}x{HEIGHT}")
app.title("Human Posture Detection")

# setting up the font
headerFont = CTkFont(family="Bebas Neue", size=50, weight="bold")
descFont = CTkFont(family="Bebas Neue", size=20)
buttonFont = CTkFont(family="Bebas Neue", size=25)

# setting up images
bgImage = Image.open("./images/MountainsECS.jpg")
background = CTkImage(light_image=bgImage,
                      dark_image=bgImage)

slouchImage = Image.open("./images/slouch.jpeg")
slouchImg = CTkImage(light_image=slouchImage,
                     dark_image=slouchImage)

yogaImage = Image.open("./images/yogaPose.jpg")
yogaImg = CTkImage(light_image=yogaImage,
                   dark_image=yogaImage)

gymImage = Image.open("./images/GymPose.jpg")
gymImg = CTkImage(light_image=gymImage,
                  dark_image=gymImage)


# functions

def button_clicked(num):
    print(f"Button {num} clicked")


def bg_resizer(e):
    if e.widget is app:
        i = CTkImage(bgImage, size=(e.width, e.height))
        bg_label.configure(text="", image=i)

# adding the background label
bg_label = CTkLabel(master=app, text="", image=background)
bg_label.place(relx=0, rely=0)


# setting up the frame

textFrame = CTkFrame(master=app, width=FRAME_WIDTH, height=TEXT_FRAME_HEIGHT, fg_color=DARK_BLUE)
textFrame.pack(expand=True)
textFrame.place(relx=0.5, rely=0.225, anchor=CENTER)

btnFrame1 = CTkFrame(master=app, width=FRAME_WIDTH, height=FRAME_HEIGHT, fg_color=LIGHT_BLUE)
btnFrame1.pack(expand=True)
btnFrame1.place(relx=0.5, rely=0.55, anchor=CENTER)

btnFrame2 = CTkFrame(master=app, width=FRAME_WIDTH, height=FRAME_HEIGHT, fg_color=BLUE)
btnFrame2.pack(expand=True)
btnFrame2.place(relx=0.5, rely=0.7, anchor=CENTER)

btnFrame3 = CTkFrame(master=app, width=FRAME_WIDTH, height=FRAME_HEIGHT, fg_color=DARK_BLUE)
btnFrame3.pack(expand=True)
btnFrame3.place(relx=0.5, rely=0.85, anchor=CENTER)


# adding the labels
heading = CTkLabel(master=textFrame, text="Human Posture Detection", font=headerFont, text_color="black")
heading.place(relx=0.5, rely=0.2, anchor=CENTER)

desc = CTkLabel(master=textFrame, text=desc_text, wraplength=FRAME_WIDTH - 50, font=descFont, justify="left")
desc.place(relx=0.5, rely=0.6, anchor=CENTER)

def runslouch():
    threading.Thread(target=slouch).start()

def runyoga():
    threading.Thread(target=yogapose).start()

def rungym():
    threading.Thread(target=gympose).start()
# adding the buttons and their backgrounds
slouchImg.configure(size=(FRAME_WIDTH, FRAME_HEIGHT + 200))
bgLabel1 = CTkLabel(master=btnFrame1, text="", image=slouchImg)
bgLabel1.place(relx=0.5, rely=1.4, anchor=CENTER)
button1 = CTkButton(master=btnFrame1, text="Posture Detection", font=buttonFont, text_color="black", fg_color=LIGHT_BLUE, command=runslouch, hover_color=DARK_BLUE)
button1.place(relx=0.5, rely=0.5, anchor=CENTER)

yogaImg.configure(size=(FRAME_WIDTH, FRAME_HEIGHT + 400))
bgLabel2 = CTkLabel(master=btnFrame2, text="", image=yogaImg)
bgLabel2.place(relx=0.5, rely=1, anchor=CENTER)
button2 = CTkButton(master=btnFrame2, text="Yoga Pose Detection", font=buttonFont, text_color="black", fg_color=LIGHT_BLUE, command=runyoga, hover_color=DARK_BLUE)
button2.place(relx=0.5, rely=0.5, anchor=CENTER)

gymImg.configure(size=(FRAME_WIDTH, FRAME_HEIGHT + 200))
bgLabel3 = CTkLabel(master=btnFrame3, text="", image=gymImg)
bgLabel3.place(relx=0.5, rely=0.5, anchor=CENTER)
button3 = CTkButton(master=btnFrame3, text="Gym Pose Detection", font=buttonFont, text_color="black", fg_color=LIGHT_BLUE, command=rungym, hover_color=DARK_BLUE)
button3.place(relx=0.5, rely=0.5, anchor=CENTER)

if __name__ == "__main__":
    app.bind("<Configure>", bg_resizer)
    app.mainloop()