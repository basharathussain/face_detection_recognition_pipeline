# Real-time Face Recognition with Auto-Registration

An advanced real-time face recognition system that automatically registers unknown faces after a timeout period. The system combines facial identification, emotion analysis, age/gender detection, and voice activity detection with intelligent auto-registration capabilities.

## 🚀 Features

### Core Recognition Features
- **Real-time Face Recognition**: Identifies known faces from a pre-built database
- **Emotion Detection**: Analyzes facial expressions (happy, sad, angry, etc.)
- **Age & Gender Estimation**: Estimates age and determines gender with confidence thresholds
- **Voice Activity Detection**: Detects when a person is speaking using audio analysis

### Auto-Registration Features
- **Unknown Face Tracking**: Automatically tracks unknown faces over time
- **Smart Face Grouping**: Groups similar unknown faces to avoid duplicates
- **Auto-Registration Timer**: Prompts for name registration after 10 seconds
- **Visual Timer Display**: Shows countdown timer for unknown faces
- **Intelligent Cleanup**: Removes stale unknown face data automatically

### Advanced Features
- **Multi-face Support**: Handles multiple faces simultaneously
- **Robust Camera Initialization**: Multiple fallback methods for camera setup
- **Audio Integration**: Optional voice activity detection
- **Frame Saving**: Save current frames with timestamp
- **Profile Management**: Automatically creates photo galleries for new profiles

## 📋 Requirements

### Dependencies

```bash
pip install opencv-python
pip install insightface
pip install deepface
pip install pyaudio
pip install webrtcvad
pip install numpy
```

### System Requirements

- **Camera**: Webcam or USB camera
- **Microphone**: For voice activity detection (optional)
- **Python**: 3.7 or higher
- **Storage**: Space for face database and photo storage

## 🛠️ Installation

### 1. Install Core Dependencies

```bash
# Install computer vision packages
pip install opencv-python insightface deepface numpy

# Install audio packages (optional)
pip install pyaudio webrtcvad
```

### 2. Windows-Specific Audio Setup

If you encounter PyAudio installation issues on Windows:

```bash
pip install pipwin
pipwin install pyaudio
```

### 3. GPU Acceleration (Optional)

For better performance with NVIDIA GPUs:

```bash
pip install onnxruntime-gpu
```

Then modify the providers in the code:
```python
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
```

## 🔧 Configuration

### Recognition Settings
```python
SIMILARITY_THRESHOLD = 0.6              # Face matching sensitivity (0.0-1.0)
MAX_DISTANCE = 1.0                      # Maximum distance for face matching
FACE_SIMILARITY_THRESHOLD = 0.8         # Threshold for grouping unknown faces
GENDER_CONFIDENCE_THRESHOLD = 0.75      # Minimum confidence for gender detection
```

### Auto-Registration Settings
```python
UNKNOWN_FACE_TIMEOUT = 10               # Seconds before prompting for registration
CAPTURE_VARIANTS = 3                    # Number of face variants to save per person
```

### Audio Settings (Optional)
```python
AUDIO_FRAME_DURATION = 30               # ms - Audio frame length
AUDIO_SAMPLE_RATE = 16000               # Hz - Audio sampling rate
AUDIO_SENSITIVITY = 1                   # 1-3 (higher = more sensitive)
```

## 🚀 Usage

### Basic Usage

```bash
python 03_realtime_rec_deepface_autoregister_unknown.py
```

### First Run

On first run, the system will:
1. Create an empty face database (`face_db.npy`)
2. Create required directories (`saved_frames/`, `employee_photos/`)
3. Initialize camera and audio systems

### Controls

- **'s' Key**: Save current frame with timestamp
- **'q' Key**: Quit the application
- **Console Input**: Enter names when prompted for unknown faces

### Auto-Registration Workflow

1. **Face Detection**: System detects unknown face
2. **Tracking**: Starts tracking the face with visual timer
3. **Grouping**: Groups similar detections of the same person
4. **Timeout**: After 10 seconds, prompts for name input
5. **Registration**: Creates profile with multiple face variants
6. **Recognition**: Future detections identify the person

## 📊 Display Information

### For Known Faces
- **Name**: Person's registered name
- **Confidence**: Recognition confidence score (0.0-1.0)
- **Age**: Estimated age
- **Gender**: Male/Female (only if confidence > 75%)
- **Emotion**: Dominant emotion
- **Speaking**: YES/no based on voice activity

### For Unknown Faces
- **Label**: "Unknown"
- **Timer**: Countdown showing time until registration prompt
- **Tracking Info**: Face ID and tracking status
- **Visual Indicators**: Red bounding box with yellow timer text

## 📁 File Structure

```
project/
├── 03_realtime_rec_deepface_autoregister_unknown.py  # Main script
├── face_db.npy                                       # Face embeddings database
├── saved_frames/                                     # Saved frame directory
│   └── frame_20241201_143022.jpg                    # Timestamped frames
├── employee_photos/                                  # Auto-generated profiles
│   ├── John_Doe/                                    # Individual profile folder
│   │   ├── variant_1.jpg                           # Face variant 1
│   │   ├── variant_2.jpg                           # Face variant 2
│   │   └── variant_3.jpg                           # Face variant 3
│   └── Jane_Smith/                                  # Another profile
└── README.md                                        # This file
```

## 🔍 System Output

### Console Output
```
🚀 Starting Enhanced Face Recognition System
Initializing camera...
✓ Camera initialized with default backend
Loading face recognition models...
✓ InsightFace model loaded
✓ Created new face database
✓ Audio system initialized
✓ System ready! Press 'q' to quit, 's' to save frame

📷 New unknown face detected: a1b2c3d4
Faces detected: 1, Unknown faces tracked: 1
⏰ Creating profile for face a1b2c3d4 (seen for 10.2s)

🔔 New unknown face detected for 10 seconds!
Enter person's name (Firstname_Lastname) or press Enter to skip: John_Doe
✓ New profile created for John_Doe with 3 images
```

### Visual Output
- **Green Boxes**: Recognized faces
- **Red Boxes**: Unknown faces
- **Yellow Timer**: Countdown for unknown faces
- **Text Overlays**: Face information and system status
- **Status Bar**: Unknown face count and instructions

## 🧠 Technical Architecture

### Core Components

1. **Face Detection**: InsightFace 'buffalo_l' model for detection and embedding
2. **Face Recognition**: Cosine similarity matching with configurable thresholds
3. **Unknown Face Tracking**: Hash-based face ID generation and similarity grouping
4. **Auto-Registration**: Time-based triggering with user input prompts
5. **Emotion Analysis**: DeepFace multi-action analysis
6. **Voice Activity**: WebRTC VAD with PyAudio integration

### Processing Pipeline

```
Camera Input → Face Detection → Known/Unknown Classification
                                        ↓
Unknown Faces → Similarity Check → Grouping → Timer → Registration Prompt
                                        ↓
Known Faces → Database Lookup → Emotion Analysis → Display Results
                                        ↓
Audio Input → Voice Activity Detection → Speaking Status
```

### Face ID Generation

```python
# Deterministic face ID from embedding
embedding_str = np.array2string(embedding, precision=6)
face_id = hashlib.md5(embedding_str.encode()).hexdigest()[:12]
```

## 🛠️ Troubleshooting

### Common Issues

**Camera Initialization Failed**
```
Error: Could not initialize camera
Solutions:
- Close other camera applications (Zoom, Teams, etc.)
- Check camera permissions
- Try running as administrator
- Test with different camera index
```

**Audio System Failed**
```
⚠ Audio setup failed: [Errno -9996] Invalid input device
Solutions:
- Check microphone permissions
- Install pyaudio correctly
- System continues without audio (speaking detection disabled)
```

**Face Database Issues**
```python
# Reset face database if corrupted
import numpy as np
face_db = {}
np.save('face_db.npy', face_db)
```

**Performance Issues**
- Reduce `det_size` for faster processing: `det_size=(320, 320)`
- Lower video resolution: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)`
- Disable emotion analysis for speed
- Use GPU acceleration if available

### Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `SystemError: new style getargs format` | OpenCV backend issue | Try different camera backends |
| `ModuleNotFoundError: insightface` | Missing dependency | `pip install insightface` |
| `Input blocked` | Name input during video loop | Enter name quickly or press Enter to skip |
| `Memory error` | Too many tracked faces | Restart application |

## ⚡ Performance Optimization

### Speed Improvements
```python
# GPU acceleration
providers=['CUDAExecutionProvider', 'CPUExecutionProvider']

# Reduced detection size
det_size=(320, 320)

# Skip emotion analysis
# Comment out analyze_gender_emotion() calls
```

### Memory Management
- Automatic cleanup of stale unknown faces after 5 seconds
- Limited frame storage per unknown face (CAPTURE_VARIANTS)
- Efficient embedding storage and comparison

### Resource Usage
- **CPU**: ~30-50% on modern processors
- **Memory**: ~200-500MB depending on face database size
- **Storage**: ~1-2MB per registered person (3 variants)

## 🔐 Privacy & Security

### Data Storage
- Face embeddings stored locally in `face_db.npy`
- Face photos stored in `employee_photos/` directory
- No cloud connectivity or external data transmission

### Privacy Features
- Local processing only
- Configurable similarity thresholds
- User control over registration process
- Automatic cleanup of temporary unknown face data

## 🎯 Use Cases

### Office/Workplace
- Employee attendance tracking
- Visitor management
- Security monitoring
- Meeting room access control

### Educational
- Student attendance
- Classroom engagement monitoring
- Campus security

### Retail/Hospitality
- Customer recognition
- VIP identification
- Staff monitoring

### Research
- Emotion analysis studies
- Behavioral research
- Human-computer interaction

## 🔄 Integration Options

### API Integration
```python
# Extract core functions for API use
from your_script import find_best_match, analyze_gender_emotion

# Use in web applications
face_info = analyze_face(image_data)
```

### Database Integration
```python
# Replace numpy storage with SQL database
import sqlite3

def save_to_database(name, embedding):
    # Your database code here
    pass
```

### Webhook Integration
```python
# Send registration events to external systems
import requests

def notify_new_registration(name, confidence):
    webhook_url = "https://your-system.com/webhook"
    requests.post(webhook_url, json={"name": name, "confidence": confidence})
```

## 📈 Future Enhancements

### Planned Features
- [ ] Web-based management interface
- [ ] Multiple camera support
- [ ] Cloud synchronization options
- [ ] Advanced analytics dashboard
- [ ] Mobile app integration

### Advanced Features
- [ ] Age progression tracking
- [ ] Mask detection
- [ ] Attendance reporting
- [ ] Integration with access control systems

## 📄 License

This project uses several open-source libraries:
- **InsightFace**: Apache 2.0 License
- **DeepFace**: MIT License
- **OpenCV**: Apache 2.0 License
- **WebRTC VAD**: BSD License

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comments for complex algorithms
- Test with multiple face types and lighting conditions
- Update documentation for new features

## 📞 Support

### Getting Help
1. Check this README for common solutions
2. Review the troubleshooting section
3. Ensure all dependencies are correctly installed
4. Test camera and microphone permissions

### Bug Reports
When reporting issues, include:
- Python version and OS
- Error messages (full traceback)
- Camera and audio hardware details
- Steps to reproduce the issue

### Feature Requests
For new features:
- Describe the use case
- Explain the expected behavior
- Consider performance implications
- Suggest implementation approach