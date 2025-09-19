import cv2
import numpy as np
import tensorflow as tf
from alert import send_fall_alert  # Make sure alert.py has send_fall_alert() defined

# Load your trained model
model = tf.keras.models.load_model("posture_fall_detection_model.h5")

# Define your label list exactly matching the number of classes your model outputs
label_list = ['fall', 'stand']  # Example 2 classes

# Open webcam stream
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to open webcam")
    exit()

def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

fall_alert_sent = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    input_img = preprocess_frame(frame)
    preds = model.predict(input_img)

    print("Raw model predictions:", preds)  # Debugging

    class_id = np.argmax(preds)
    confidence = preds[0][class_id]

    if class_id >= len(label_list):
        label = "Unknown"
        print(f"Warning: Predicted class_id {class_id} out of range")
    else:
        label = label_list[class_id]

    display_text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, display_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Fall Detection", frame)

    # If fall detected and confidence high enough, send alert once
    if label == 'fall' and confidence > 0.8 and not fall_alert_sent:
        send_fall_alert("number")  # Replace with real recipient number
        fall_alert_sent = True
        print("Fall alert sent!")

    # Reset alert flag when detection changes to non-fall
    if label != 'fall':
        fall_alert_sent = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
