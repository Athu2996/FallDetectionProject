# Fall Detection Project

A **real-time posture and fall detection system** using deep learning and computer vision.  
The system uses a webcam to detect if a person has fallen, and if so, it sends **SMS alerts via Twilio** to caregivers.

---

## 🚀 Project Overview

- Captures live video using **OpenCV**.
- Uses a trained **TensorFlow/Keras model** to classify posture (`fall` or `stand`).
- Sends an **SMS alert** to a caregiver when a fall is detected (via Twilio API).
- Runs in real time on a local machine.

---

## 📂 Project Structure

allDetectionProject/
├── dataset/ # Training & validation data
│ ├── images/ # Image dataset
│ └── labels/ # Labels for training
├── main.py # Script to train and save the model
├── realtime_inference.py # Real-time webcam inference & alert
├── alert.py # Twilio SMS alert module
├── posture_fall_detection_model.h5 # Saved trained model
├── requirements.txt # Python dependencies
└── README.md # Project documentation



Create a Twilio account
.

Get your Account SID, Auth Token, and a Twilio phone number.

Verify caregiver phone numbers if you’re on a Twilio trial account.

Open alert.py and update:

account_sid = "your_account_sid"
auth_token = "your_auth_token"
twilio_number = "+1234567890"  # Your Twilio number
caregiver_number = "+919999999999"  # Replace with caregiver’s number

🏋️ Training the Model (Optional)

If you want to train the model with your dataset:

python main.py


A new model file posture_fall_detection_model.h5 will be created.

🎥 Run Real-Time Detection
python realtime_inference.py


A webcam window will open.

Predictions (fall or stand) will appear on the video feed.

If a fall is detected with >80% confidence, an SMS alert is sent.

Press q to quit.



Explanation:
tensorflow and keras for deep learning model handling

numpy for numerical operations

pandas for any dataset manipulation if needed

opencv-python for realtime video capture and processing

twilio for SMS alert integration


