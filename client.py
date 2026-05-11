
import requests
import os

url = "http://localhost:8000/predict"
image_path = r"c:\Users\USER\Documents\main_cleaned\seg_test\seg_test\buildings\20057.jpg"  # Use a known valid image

if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit()

with open(image_path, "rb") as file:
    files = {"file": (os.path.basename(image_path), file, "image/jpeg")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code} - {response.text}")
