import requests

# This is just a fake alert to test connection
alert_data = {
    "weaponType": "pistol",
    "confidence": 0.92,
    "cameraId": "test_cam",
    "imageUrl": None
}

try:
    res = requests.post("http://localhost:5000/api/alerts", json=alert_data)
    print("Status:", res.status_code)
    print("Response:", res.json())
except Exception as e:
    print("❌ Error:", e)
