import os
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torch import nn
from torch.utils.data import Dataset
import face_recognition
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = 'e9f3e8a7d5c1f8b1a2d3c4e5f6g7h8i9'

ALLOWED_VIDEO_EXTENSIONS = set(['mp4', 'avi', 'mov', 'mkv'])
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.linear1 = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x)
        return fmap, self.linear1(x_lstm[:, -1, :])

class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []

        # Extract frames from the video
        for frame in self.frame_extract(video_path):
            if len(frames) < self.sequence_length:
                transformed_frame = self.transform(frame)
                frames.append(transformed_frame)

        # If there are fewer frames than sequence_length, pad with zeros
        while len(frames) < self.sequence_length:
            frames.append(torch.zeros_like(frames[0]))  # Pad with zeros (C, H, W)

        frames = torch.stack(frames)[:self.sequence_length]  # Stack to create a tensor
        return frames.unsqueeze(0)  # Add batch dimension

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        frame_count = 0
        
        while True:
            success, image = vidObj.read()
            if not success:
                break
            
            # Convert BGR to RGB and check image size
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[0] == 0 or image.shape[1] == 0:  # Check if the image is valid
                continue
            
            frame_count += 1
            yield image

        print(f"Total frames extracted from video: {frame_count}")

# Load model and weights
model = Model(2).to(device)
model.load_state_dict(torch.load('models/model_84_acc_10_frames_final_data.pt', map_location=device, weights_only=True))

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def predict(model, frames):
    model.eval()
    with torch.no_grad():
        fmap, logits = model(frames.to(device))
        logits = sm(logits)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = logits[0][prediction].item() * 100
    return prediction, confidence

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url) 
        
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url) 
        
        if not allowed_video_file(file.filename):
            return redirect(request.url) 
        
        # Save the video file
        video_path = os.path.join('static/uploaded_videos', file.filename)
        file.save(video_path)

        session['video_path'] = video_path
        return redirect(url_for('predict_page')) 
        
    return render_template('detect.html')

def extract_faces_from_frame(frame):
    # Convert frame from BGR (OpenCV) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    faces = []
    for top, right, bottom, left in face_locations:
        face_image = rgb_frame[top:bottom, left:right]  # Extract the face from the frame
        face_image = cv2.resize(face_image, (112, 112))  # Resize to desired dimensions
        faces.append(face_image)

    return faces

@app.route('/predict')
def predict_page():
    video_path = session.get('video_path')
    if not video_path:
        return redirect(url_for('home'))

    # Initialize variables
    extracted_frames = []
    extracted_faces = []
    
    # Ensure directories exist for saving images
    os.makedirs('static/extracted_frames', exist_ok=True)
    os.makedirs('static/extracted_faces', exist_ok=True)

    # Open video for frame extraction
    vidObj = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = vidObj.read()
        if not success or frame_count >= 10:  # Stop after 10 frames
            print(f"Finished processing video. Total frames processed: {frame_count}")
            break
        frame_count += 1
        
        # Transform the frame
        transformed_frame = train_transforms(frame)

        # Ensure transformation succeeded
        if transformed_frame is not None and transformed_frame.shape[0] == 3:
            # Save the original frame
            frame_path = f"static/extracted_frames/frame{frame_count}.jpg"
            extracted_frames.append(frame_path)
            cv2.imwrite(frame_path, frame)

            # Extract faces from the transformed frame
            faces = extract_faces_from_frame(frame)
            if faces:  # Check if any faces were detected
                for j, face in enumerate(faces):
                    face_path = f"static/extracted_faces/face{frame_count}_{j}.jpg"
                    cv2.imwrite(face_path, face)  # Save each face image
                    extracted_faces.append(face_path)
            else:
                print(f"No faces found in frame {frame_count}.")
        else:
            print(f"Failed to transform frame {frame_count}")

    # Perform prediction if enough frames are extracted
    if len(extracted_frames) >= 10:
        frames_tensor = torch.stack([train_transforms(cv2.imread(frame)) for frame in extracted_frames]).unsqueeze(0)
        prediction, confidence = predict(model, frames_tensor)
        result = "REAL" if prediction == 1 else "FAKE"
    else:
        result = "UNKNOWN"
        confidence = 0

    # Debug print statements
    print(f"Extracted frames: {extracted_frames}")
    print(f"Extracted faces: {extracted_faces}")

    return render_template('predict.html', result=result, confidence=confidence,
                           extracted_frames=extracted_frames[:10],  # Only show 10 frames
                           extracted_faces=extracted_faces[:10])  # Only show 10 faces




if __name__ == '__main__':
    app.run(debug=True)
