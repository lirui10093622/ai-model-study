import requests
from PIL import Image, ImageDraw
from transformers import pipeline

url = "https://unsplash.com/photos/oj0zeY2Ltk4/download?ixid-MnwxMjA3fDB8MXxzZMFyY2h8MTR8fHpY25Y3xlbnwwfHx8fDE2Nzc0OTE1NDk&force=true&w=640"
image = Image.open(requests.get(url, stream=True).raw)

checkpoint = "google/owlvit-base-patch32"
detector = pipeline("zero-shot-object-detection", checkpoint)
predictions = detector(image=image, candidate_labels=["hat", "sunglasses"])
print(predictions)

draw = ImageDraw.Draw(image)

for prediction in predictions:
    box = prediction["box"]
    label = prediction["label"]
    score = prediction["score"]
    xmin, ymin, xmax, ymax = box.values()

    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
    draw.text((xmin, ymin), f"{label}: {round(score, 2)}", fill="red")

image.show()
