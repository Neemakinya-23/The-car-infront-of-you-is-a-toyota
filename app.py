from collections import Counter
from pathlib import Path

import cv2
from flask import Flask, render_template_string, request, send_from_directory, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOADS = Path("uploads")
UPLOADS.mkdir(exist_ok=True)
model = YOLO("best.pt")

HTML = """
<!doctype html>
<html>
<head>
	<title>Logo Predictor</title>
</head>
<body style="font-family: Arial; max-width: 700px; margin: 40px auto; line-height: 1.6;">
	<h2>Upload an Image or Video</h2>
	<form method="post" enctype="multipart/form-data">
		<input type="file" name="file" required>
		<button type="submit">Predict</button>
	</form>

	{% if message %}
		<p><b>{{ message }}</b></p>
	{% endif %}

	{% if file_url and is_video %}
		<video controls style="max-width: 100%; margin-top: 12px;" src="{{ file_url }}"></video>
	{% elif file_url %}
		<img style="max-width: 100%; margin-top: 12px;" src="{{ file_url }}" alt="Uploaded file">
	{% endif %}
</body>
</html>
"""


def predict_label(file_path):
		ext = file_path.suffix.lower()
		if ext in {".mp4", ".avi", ".mov", ".mkv"}:
				return predict_video(file_path)
		result = model.predict(source=str(file_path), task="classify", verbose=False)[0]
		probs = result.probs
		if probs is None:
				return "No prediction", 0.0
		scores = probs.data
		top1 = int(scores.argmax().item())
		return result.names[top1], float(scores[top1].item())


def predict_video(file_path):
		cap = cv2.VideoCapture(str(file_path))
		labels = []
		frame_idx = 0
		while True:
				ok, frame = cap.read()
				if not ok:
						break
				if frame_idx % 15 == 0:  # sample every 15th frame
						result = model.predict(source=frame, task="classify", verbose=False)[0]
						probs = result.probs
						if probs is not None:
								scores = probs.data
								top1 = int(scores.argmax().item())
								labels.append(result.names[top1])
				frame_idx += 1
		cap.release()
		if not labels:
				return "No prediction", 0.0
		top_label = Counter(labels).most_common(1)[0][0]
		confidence = labels.count(top_label) / len(labels)
		return top_label, confidence


@app.route("/uploads/<path:filename>")
def uploads(filename):
		return send_from_directory(UPLOADS, filename)


@app.route("/", methods=["GET", "POST"])
def index():
		message = ""
		file_url = ""
		is_video = False

		if request.method == "POST":
				file = request.files.get("file")
				if not file or not file.filename:
						message = "Please choose a file."
				else:
						filename = secure_filename(file.filename)
						saved_path = UPLOADS / filename
						file.save(saved_path)
						file_url = url_for("uploads", filename=filename)

						label, confidence = predict_label(saved_path)
						is_video = saved_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
						message = f"Predicted brand: {label} ({confidence:.2%})"

		return render_template_string(HTML, message=message, file_url=file_url, is_video=is_video)


if __name__ == "__main__":
		app.run(debug=True)
