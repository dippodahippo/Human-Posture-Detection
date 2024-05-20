import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
import pyduinointegr.pyduino_connection as pyd
def main():
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
        global prev_prediction
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