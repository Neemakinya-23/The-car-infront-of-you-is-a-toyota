# The-car-infront-of-you-is-a-toyota

This is a small and fun YOLOv8 project.

It takes in different car logos and predicts their brand. I enjoyed training the model with my own data and seeing it make predictions.

## Project Structure

- `logo_det/` - logo images and labels used for training
- `outputs/` - prediction outputs (this is also where you can find trained weights such as `best.pt`)
- `uploads/` - newly uploaded images and videos from the web app
- `app.py` - Flask web application

## Why I Built This

I wanted to get a better understanding of YOLOv8 while manifesting a Mercedes in 2026, and to test the idea that "every car in front of you is indeed a Toyota".

## Run the App

1. Open PowerShell in this project folder.
2. Activate the virtual environment:

```powershell
& ..\yolo_v8_env\Scripts\Activate.ps1
```

3. Install dependencies (first time only):

```powershell
pip install flask ultralytics opencv-python
```

4. Start the app:

```powershell
python .\app.py
```

5. Open your browser and go to:

`http://127.0.0.1:5000`

## Demo Recording

Project recording:

[Recording 2026-03-16](./Recording%202026-03-16%20230418.mp4)

Initial project commit.
