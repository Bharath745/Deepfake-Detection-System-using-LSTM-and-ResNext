# Deepfake-Detection-System-using-LSTM-and-ResNext

This project is a Deepfake Detection System developed using Flask. It combines the strengths of LSTM (Long Short-Term Memory) and ResNeXt models to analyze video files and detect manipulations. The system leverages sequential and spatial analysis of video frames to determine if a video is real or fake.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

Deepfake videos are increasingly used for deceptive purposes, making it crucial to develop systems to detect them. This project provides a framework to upload and analyze videos for deepfake detection, using LSTM for temporal (sequence-based) analysis and ResNeXt for spatial (frame-based) analysis. By examining video frames and detecting irregularities, this system can determine if a video is likely manipulated.

## Features

- **Upload Video**: Allows users to upload video files for analysis.
- **Detection Results**: Displays analysis results with confidence scores.
- **Extracted Frames and Faces**: Provides extracted frames and facial regions used for detection, presented in a user-friendly carousel format.
- **Real-Time Playback**: Allows users to preview and playback uploaded videos.

## Technologies Used

- **Flask**: Backend web framework to manage routes and handle video uploads.
- **LSTM**: For analyzing temporal inconsistencies across video frames.
- **ResNeXt**: For detecting spatial anomalies in individual frames.
- **HTML/CSS/JavaScript**: For creating an interactive and responsive user interface.

## Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

- Python 3.8+
- Flask
- OpenCV
- Other dependencies in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/bharath745/deepfake-detection-system.git](https://github.com/Bharath745/Deepfake-Detection-System-using-LSTM-and-ResNext.git)
   cd /Deepfake-Detection-System-using-LSTM-and-ResNext
   
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   
3. Run the Flask application:
   ```bash
   python app.py

4. Open a web browser and navigate to http://127.0.0.1:3000/ to access the application.

### Project Structure

```plaintext
Deepfake-Detection-System-using-LSTM-and-ResNext/
│
├── app/             
│   ├── static/
│   │   ├── css/                  # CSS files
│   │   ├── images/               # images used in project
│   │   ├── Extracted faces/      # images of Extracted faces 
│   │   ├── Uploaded videos/      # videos uploaded in project
├── templates/                    # HTML templates for Flask views
├── models/                       # Contains LSTM and ResNeXt models    
├── app.py                        # Main server file to run the Flask application
├── requirements.txt              # Dependencies for the project
└── README.md                     # Project README file
```
### Usage

1. **Home Page**: Upon visiting the application, users will be presented with the option to upload a video.
2. **Upload Video**: Users can select a video file and upload it for detection.
3. **Detection Results**: After uploading, the system processes the video and displays results, showing whether the video is real or fake along with confidence percentages.
4. **View Extracted Frames**: Users can see extracted frames and facial regions from the video, aiding in visual understanding of the detection process.

### Screenshots
#### Home Page
![image](https://github.com/user-attachments/assets/88a5b63b-1e29-4b9c-bb1f-b24c6138c7a2)


#### Video Upload Page
![image](https://github.com/user-attachments/assets/a188c6fb-994c-45ea-b578-e7e0ec1da8a9)


#### Result Page
![image](https://github.com/user-attachments/assets/41d4b09c-93b7-4bcd-b2dd-706d408b2fcd)

### Future Work
In future iterations, we aim to:

1. **Incorporate Review Content Analysis**: Integrate user feedback and content analysis for deeper insights.
2. **Explore Social Network Data**: Enhance detection accuracy by extracting relevant social network data.
3. **Improve Model Efficiency**: Optimize the system for real-time deepfake detection capabilities.

### License
This project is licensed under the MIT License. See the `LICENSE` file for details.

