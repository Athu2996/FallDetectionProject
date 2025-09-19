from twilio.rest import Client
import os

# Enter your Twilio credentials here or set them as environment variables
# You can get these by signing up at https://www.twilio.com/console
ACCOUNT_SID = 'your key'
AUTH_TOKEN = 'your key'
TWILIO_PHONE = 'number'  # Your Twilio phone number

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_fall_alert(phone_number):
    try:
        message = client.messages.create(
            body="Warning: Fall detected! Immediate assistance required.",
            from_=TWILIO_PHONE,
            to=phone_number
        )
        print(f"Alert sent with SID: {message.sid}")
    except Exception as e:
        print(f"Error sending alert: {e}")

# For testing: call alert function when this script is run directly
if __name__ == "__main__":
    # Replace below with your own number in international format
    send_fall_alert("number")
