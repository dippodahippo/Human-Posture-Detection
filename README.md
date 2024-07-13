# ML Based Posture Recognition System
The aim of our project is to address and correct posture issues in individuals.The primary goal is to help people maintain proper alignment and posture in various scenarios such as sitting, performing yoga poses, and during gym exercises.

# Table of contents
- [Introduction](#introduction)
- [Resources](#resources)
- [Data Collection and Augmentation](#datacollection)
- [Model training Based on data Collection](#modeltraining)
- [Arduino wrist band designing,building](#arduinowristband)
- [Hardware Software integration](#hardwareintegration)
- [Testing with real time video capture](#testing)
- [Installation](#installation)
- [Usage via web based interface](#interface)
- [Scope for Future Improvement and Scalability](#futurescope)

  
# Introduction:

In today's increasingly sedentary lifestyle, maintaining good posture has become more critical than ever. Poor posture not only leads to physical discomfort but also increases the risk of long-term musculoskeletal problems, affecting overall well-being and quality of life. Recognizing the profound impact of posture on health, our project aims to provide comprehensive support and guidance to individuals striving to maintain good posture during various activities, such as sitting, yoga, and gym exercises.

Our project's goal is to develop an innovative solution that promotes and sustains proper posture habits. By integrating advanced technologies and user-friendly features, we aspire to assist users in reducing the risk of posture-related issues, enhancing physical health, and improving overall well-being. Our approach encompasses real-time posture monitoring, personalized feedback, and practical recommendations tailored to individual needs.

Whether sitting at a desk, practicing yoga, or engaging in gym workouts, our solution offers continuous support to ensure that users maintain optimal posture throughout their activities. By fostering awareness and encouraging corrective measures, we aim to prevent the adverse effects of poor posture and contribute to a healthier, more active lifestyle.

Join us in our mission to transform posture health and empower individuals to achieve their best physical self through informed and supported postural practices.

# Resources
## Software Libraries
### MediaPipe
MediaPipe is used for real-time pose estimation. It tracks and analyzes the user's body posture during various activities, such as sitting, yoga, and gym exercises. By detecting key points on the body, MediaPipe helps in identifying improper postures and provides data for corrective feedback.

### OpenCV
OpenCV is utilized for image and video analysis. It processes the input from cameras or videos to detect and recognize the user's body movements. OpenCV works in tandem with MediaPipe to enhance the accuracy of posture detection and to preprocess the video feed, such as background subtraction and noise reduction.

### TensorFlow & Keras
TensorFlow and Keras are used for developing and training machine learning models that classify and predict posture quality. These models analyze the data provided by MediaPipe and OpenCV to determine if the user's posture is correct or needs adjustment. TensorFlow's deep learning capabilities enable the creation of robust models that improve over time with more data.

### Customized Version of C++ for Arduino
The customized version of C++ for Arduino is used to develop firmware for hardware components, such as wearable devices or posture correction gadgets. These devices provide haptic or visual feedback to the user when an incorrect posture is detected. Arduino's efficient handling of limited memory and processing power makes it suitable for real-time posture correction applications.
## Hardware Components
### Arduino
Arduino provides a user-friendly platform for programming and controlling hardware components like LEDs. Its simple IDE and beginner-friendly programming language make it accessible even to those with limited electronics experience.
In our project, Arduino is used to control the hardware components of the posture correction device. It processes input from sensors and provides feedback through LEDs or other actuators. Arduino's ease of use allows for quick prototyping and iteration of the posture correction system
### Type A to Type B USB Connector
The Type A to Type B USB connector is used to establish a connection between the posture correction wristband and the computer. This connection allows for data transfer between the wristband sensors and the software running on the computer, enabling real-time posture monitoring and feedback.
### LEDs
In our project, LEDs serve as visual indicators for the user. One LED lights up to confirm that data from the sensors is being received correctly. The other LED provides feedback on the user's posture, turning on or changing color to signal when the posture is correct or needs adjustment. This immediate visual feedback helps users make necessary corrections in real-time.

# DataCollection
### Gym Pose Dataset
To develop a robust gym pose detection model, we collected a dataset of 280 gym pose images. Each image captures both the initial and final stages of various gym poses. These images were labeled by the specific pose they depict. 

To enhance the dataset and improve the model's generalization, we used the Python library `imgaug` for data augmentation. Each image was augmented twelve times using the following techniques:
- Tilting (5° left and right)
- Adding Gaussian noise
- Blurring
- Distorting
- Flipping horizontally

This augmentation process resulted in a dataset of 3360 unique samples, significantly expanding the original dataset and providing diverse training data for the model.

### Yoga Pose Dataset
For the yoga pose detection model, we started with a collection of 306 images. These images were labeled by pose, with each sequence of images representing a specific yoga pose (e.g., images 1-13 show Anjaneyasana, 14-25 show Virabhadrasana, etc.).

To augment this dataset, we again used the `imgaug` library. Each image was augmented six times using the following techniques:
- Tilting (5° left and right)
- Adding Gaussian noise
- Blurring
- Distorting
- Flipping horizontally

This augmentation resulted in a dataset of 2142 unique samples, increasing the variety of training examples and enhancing the model's ability to accurately detect different yoga poses.

### Slouch Detection
The slouch detection system operates in real-time and does not rely on any predefined model or dataset. Consequently, there was no need for data augmentation for this aspect of the project. The system uses live data to detect and correct slouching, providing immediate feedback to the user.

# ModelTraining

To develop our pose detection models, we employed a Convolutional Neural Network (CNN) architecture consisting of 4 convolutional layers and 3 dense layers. Additional layers, such as flattening and MaxPooling2D, were incorporated to create a complex neural connection between the input images, allowing the model to identify patterns in the testing data, which is real-time data.

### Model Architecture

The following code snippet demonstrates the architecture of the CNN used for both the gym pose and yoga pose detection models:

```python
import tensorflow as tf

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((33, 2, 1), input_shape=(33, 2)),  # Reshape the input data
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),  # First Conv2D layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),  # Second Conv2D layer
    tf.keras.layers.MaxPooling2D((2, 2)),  # MaxPooling2D layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  # Third Conv2D layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  # Fourth Conv2D layer
    tf.keras.layers.Flatten(),  # Flatten layer
    tf.keras.layers.Dense(128, activation='relu'),  # First Dense layer
    tf.keras.layers.Dense(64, activation='relu'),  # Second Dense layer
    tf.keras.layers.Dense(21, activation='softmax')  # Output Dense layer with 21 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(poses, labels, epochs=100, batch_size=32)
```

### Explanation of Layers

- **Reshape Layer:** Reshapes the input data to match the required input shape for the convolutional layers.
- **Conv2D Layers:** Apply convolution operations with 32 and 64 filters, using ReLU activation and 'same' padding. These layers help the model to learn spatial hierarchies in the images.
- **MaxPooling2D Layer:** Reduces the spatial dimensions of the data, which helps in down-sampling and reduces the computational complexity.
- **Flatten Layer:** Flattens the 2D matrix data into a 1D vector, preparing it for the dense layers.
- **Dense Layers:** Fully connected layers with ReLU activation. The final dense layer uses a softmax activation function to output probabilities for each class (pose).

### Model Training

The model is compiled using the Adam optimizer, `sparse_categorical_crossentropy` loss function, and accuracy as the evaluation metric. It is then trained on the augmented dataset for 100 epochs with a batch size of 32.

### Real-Time Pose Detection

As the user attempts poses in front of the camera, the live feed is detected by OpenCV using the system’s webcam. OpenCV, a popular computer vision library in Python, captures the live video stream. This information is then fed to MediaPipe, another well-known image detection library, to extract keypoints of the person, such as major joints. These keypoints are connected and compared to the trained model, which evaluates and displays how accurately the poses are executed.


# ArduinoWristband
The Arduino wristband is a versatile and user-friendly device designed to aid individuals in maintaining proper posture during various activities. The wristband features two LEDs, each serving a distinct purpose to enhance user experience and functionality. LED1 acts as a data input indicator, blinking continuously during real-time video capture to reassure the wearer that data is actively being collected. This visual cue boosts confidence in the device's operation.
![image](https://github.com/user-attachments/assets/8c4b07de-554b-4d0c-9959-8343320b5420)

Meanwhile, LED2 functions as an output feedback indicator, alerting the user to instances of inaccurate posture. Through integrated sensors and algorithms, the wristband detects deviations from the desired posture, prompting LED2 to blink and notifying the wearer to make necessary adjustments. This immediate feedback mechanism is crucial for real-time posture correction.

The wristband is built using an Arduino board, which controls the LEDs and processes input from the sensors. A Type A to Type B USB connector links the wristband to a computer for data transfer and power supply. The Arduino is programmed to control the LEDs based on posture data processed by OpenCV and MediaPipe, which analyze live video feed from the system’s webcam. This seamless integration of hardware and software ensures that users receive prompt and accurate feedback, helping them to maintain proper posture and reduce the risk of posture-related issues.

```python
const int led = LED_BUILTIN;
char msg;


void setup() {
  Serial.begin(9600);
  pinMode(led, OUTPUT);
  digitalWrite(led, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    msg = Serial.read();
    Serial.println(msg);
    
    if (msg == '1'){
      digitalWrite(LED_BUILTIN, HIGH);
      delay(500);
    }

    else if (msg == '0') {
      digitalWrite(led, LOW);
      for(int i = 0; i < 5; i++){
        digitalWrite(led, HIGH);
        delay(100);
        digitalWrite(led, LOW);
        delay(100);
        }
    }

    else {
      digitalWrite(led, LOW);
      delay(10000);
    }
  } else {
    digitalWrite(led, LOW);
  }

}
```
# HardwareIntegration

To integrate the software and hardware modules of our posture correction system, we employed a Type A to Type B USB connector. This connector serves as a link between the Arduino wristband and the computer, enabling data transfer and power supply. 

Our software development utilized a combination of the Arduino IDE and Python. The main project was developed in Python, where we facilitated communication between Python and Arduino using the PySerial library. PySerial enables serial communication between Python and Arduino, allowing seamless interaction between the Python codebase and the Arduino hardware.

This setup ensures effective coordination and control between the software and hardware components. The Python code captures real-time video feed using OpenCV, processes it with MediaPipe to extract keypoints and analyze posture, and then sends the posture data to the Arduino via the USB connection. The Arduino, in turn, controls the LEDs based on the received posture data, providing immediate feedback to the user. This integration is crucial for the real-time functionality of the posture correction system, ensuring that users receive prompt and accurate alerts to maintain proper posture.

# Testing
## Slouch Detection 
![image](https://github.com/user-attachments/assets/6511a3cf-4edb-49b5-a7a9-c794702c408c)
### Figure 1: Output screen when posture is correct

![image](https://github.com/user-attachments/assets/1310ecd5-7be0-4c8d-9ebd-eed5077c0c87)
### Figure 2: Output screen when slouching

## Yoga Pose Detection
![image](https://github.com/user-attachments/assets/40da9715-4792-48e2-a50c-b97c8b52c5a9)
### Figure 3: Detecting Vrikshasana
![image](https://github.com/user-attachments/assets/75923ee5-a8e7-43bb-b345-c421f9796067)
### Figure 4: Detecting Ardha Chakrasana
## Gym Pose Detection
![image](https://github.com/user-attachments/assets/07fd0491-3d4c-4aa3-918a-025ba903a3b5)
### Figure 5: Gym pose model detecting shoulder press down position
![image](https://github.com/user-attachments/assets/0e3321b6-46c7-4897-916c-c156135687e8)
### Figure 6: Gym pose model recalculating accuracy on the way up

# Installation
1. **Clone the Repository:**
   Open your terminal or command prompt and navigate to the directory where you want to clone the repository. Then, execute the following command:

   ```
   git clone https://github.com/dippodahippo/Human-Posture-Detection
   ```


2. **Navigate to the Cloned Directory:**
   After cloning, navigate into the cloned directory:

   ```
   cd <repository-directory>
   ```

   Replace `<repository-directory>` with the name of the directory where the repository was cloned.

3. **Install Required Packages:**
   Install all the required packages in your system which are mentioned in resouces section

   ```
   pip install -r <package-name>
   ```

   This command installs `<package-name>` package

4. **Verify Installation:**
   After installing, verify if everything is set up correctly. Depending on the repository's instructions, there might be additional setup steps such as configuring environment variables, database migrations, etc.

These steps should help you get started with cloning a repository and installing its required packages.

# Interface
1. **Navigate to the Application Directory:**
   Open your terminal or command prompt and navigate to the directory where `app.py` is located, as before:

   ```
   cd <repository-directory>
   ```

2. **Activate Virtual Environment (if applicable):**
   If you're using a virtual environment, activate it as mentioned earlier:

   ```
   source <virtual-env>/bin/activate
   ```

3. **Run the Application:**
   Start the application using Python:

   ```
   python app.py
   ```

4. **Automatically Opened Pop-up:**
   After executing `python app.py`, the application should automatically open a pop-up window or tab in your default web browser. This pop-up will display the web interface where you can interact with the application.

   ![image](https://github.com/user-attachments/assets/5933d88c-4082-41a2-9bfc-44494c84a2e2)


6. **Choose Detection Type:**
   On the pop-up interface, you should see options to choose among yoga, gym, and slouch detection. Simply click on the option you want to use.

7. **Interact with the Application:**
   Proceed to interact with the application through the pop-up interface as intended by its design and functionality.

# FutureScope

### Hardware Enhancement:

The current implementation of this device operates in wired mode and utilizes an Arduino UNO microcontroller. However, to overcome processing speed limitations and enable real-time updates of images, future iterations can benefit from integrating a more powerful microchip on a custom-built PCB. This upgrade will significantly enhance the device's performance, allowing for smoother operation and improved responsiveness.

### Dataset Expansion:

Enhancing the accuracy of the image processing model can be achieved by incorporating a larger dataset comprising tens of thousands of training images. This expansion will bolster the model's robustness and accuracy, providing a more reliable solution for real-world applications.

### Real-World Applications:

Once integrated with the developed software, this device holds promising applications in various real-world scenarios. It aims to revolutionize exercise routines by offering real-time feedback and guidance, thereby reducing the risk of injuries and promoting a healthier lifestyle. Potential applications include:

- **Fitness Monitoring:** Providing detailed insights into exercise form and posture.
- **Physical Therapy:** Assisting in rehabilitation exercises with precise feedback.
- **Sports Training:** Enhancing coaching by analyzing and improving athletic movements.

### Open-Source Collaboration:

This repository encourages collaboration and contributions from the community. Developers, researchers, and enthusiasts can explore, extend, and enhance the device's capabilities, fostering innovation in wearable technology and computer vision applications.

## Authors
- [@Diptorshi Tripati](https://www.github.com/dippodahippo)
- [@Rishika Ghosh](https://www.github.com/)
- [@Srujana Gayatri Chaganti](https://www.github.com/5rujana)
- [@Raj Aryan Das](https://www.github.com/raj-dash)
- [Usyad](https://www.github.com/)
- [Snehadipa Mukharjee](https://www.github.com/)

### The images were all procured by our team and are not for public use.
