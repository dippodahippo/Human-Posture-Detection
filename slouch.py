import cv2
import mediapipe as mp
import pyduinointegr.pyduino_connection as pyd
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

def main():
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
if __name__=="__main__":
    main()
