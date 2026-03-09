from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pygame
import math
from datetime import datetime
import csv
import psutil
import os
import urllib.request
import bz2
import shutil
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tkinter as tk
from tkinter import ttk
import threading
import io
import telegram

def download_file(url, filename):
    print(f"Downloading {filename}...")
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename} successfully!")

def check_and_download_files():
    # Check and download haarcascade file
    cascade_file = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_file):
        cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        download_file(cascade_url, cascade_file)
    
    # Check and download shape predictor file
    predictor_file = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_file):
        predictor_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        bz2_file = predictor_file + ".bz2"
        
        # Download the compressed file
        download_file(predictor_url, bz2_file)
        
        # Decompress the file
        print("Decompressing shape predictor file...")
        with bz2.BZ2File(bz2_file) as fr, open(predictor_file, 'wb') as fw:
            shutil.copyfileobj(fr, fw)
        
        # Remove the compressed file
        os.remove(bz2_file)
        print("Shape predictor file ready!")

# Initialize variables
ear = 0
distance = 0
gaze_angle = 0
horizontal_tilt = 0

# Initialize alarm control variables
alarm_status = False
alarm_status2 = False
alarm_status3 = False
saying = False
COUNTER = 0
paused = False

# Initialize metrics dictionary
performance_metrics = {
    "drowsiness_events": 0,
    "yawn_count": 0,
    "gaze_away_count": 0,
    "head_tilt_events": 0,
    "start_time": time.time(),
    "frame_count": 0,
    "last_fps_update": time.time(),
    "fps": 0,
    "data": {
        "time": [],
        "ear": [],
        "yawn": [],
        "head_tilt": []
    }
}

# Function to update metrics
def update_metrics(metrics, ear, yawn, head_tilt, event_type=None):
    """Update metrics with new values and track events"""
    current_time = time.time()
    metrics["duration"] = current_time - metrics["start_time"]
    
    # Update data points
    metrics["data"]["time"].append(metrics["duration"])
    metrics["data"]["ear"].append(ear)
    metrics["data"]["yawn"].append(yawn)
    metrics["data"]["head_tilt"].append(head_tilt)
    
    # Track events if event_type is provided
    if event_type:
        if event_type == "drowsiness":
            metrics["drowsiness_events"] += 1
        elif event_type == "yawn":
            metrics["yawn_count"] += 1
        elif event_type == "gaze_away":
            metrics["gaze_away_count"] += 1
        elif event_type == "head_tilt":
            metrics["head_tilt_events"] += 1
    
    # Keep only last 1000 data points for memory efficiency
    if len(metrics["data"]["time"]) > 1000:
        for key in metrics["data"]:
            metrics["data"][key] = metrics["data"][key][-1000:]

# Customization Settings
custom_settings = {
    # Detection Thresholds
    "eye_ar_threshold": 0.3,
    "eye_ar_consecutive_frames": 30,
    "yawn_threshold": 20,
    "horizontal_tilt_threshold": 10,
    "vertical_tilt_threshold": 10,
    "gaze_threshold": 30,
    "blink_rate_threshold": 20,  # blinks per minute
    
    # Alarm Settings
    "alarm_volume": 50,  # 0-100
    "alarm_duration": 2,  # seconds
    "alarm_repeat_interval": 10,  # seconds
    
    # UI Settings
    "font_size": 0.7,
    "text_color": (0, 255, 0),
    "alert_color": (0, 0, 255),
    "theme": "dark",  # dark/light
    
    # Notification Settings
    "telegram_alerts": True,
    "email_alerts": False,
    "sound_alerts": True,
    "visual_alerts": True,
    
    # Break Settings
    "break_interval": 60,  # minutes
    "break_duration": 5,   # minutes
    "break_reminder": True
}

def save_settings(settings, filename="settings.json"):
    """Save settings to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(settings, f, indent=4)
        print("Settings saved successfully")
    except Exception as e:
        print(f"Error saving settings: {str(e)}")

def load_settings(filename="settings.json"):
    """Load settings from JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return custom_settings  # Return default settings if file doesn't exist
    except Exception as e:
        print(f"Error loading settings: {str(e)}")
        return custom_settings

def update_thresholds():
    """Update detection thresholds from settings"""
    global EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES, YAWN_THRESH
    global HORIZONTAL_TILT_THRESH, VERTICAL_TILT_THRESH, GAZE_THRESH
    
    EYE_AR_THRESH = custom_settings["eye_ar_threshold"]
    EYE_AR_CONSEC_FRAMES = custom_settings["eye_ar_consecutive_frames"]
    YAWN_THRESH = custom_settings["yawn_threshold"]
    HORIZONTAL_TILT_THRESH = custom_settings["horizontal_tilt_threshold"]
    VERTICAL_TILT_THRESH = custom_settings["vertical_tilt_threshold"]
    GAZE_THRESH = custom_settings["gaze_threshold"]

def display_settings_menu():
    """Show settings menu in a separate window"""
    def save_and_close():
        # Update settings dictionary with new values
        custom_settings["eye_ar_threshold"] = float(ear_entry.get())
        custom_settings["eye_ar_consecutive_frames"] = int(ear_frames_entry.get())
        custom_settings["yawn_threshold"] = int(yawn_entry.get())
        custom_settings["horizontal_tilt_threshold"] = int(h_tilt_entry.get())
        custom_settings["vertical_tilt_threshold"] = int(v_tilt_entry.get())
        custom_settings["gaze_threshold"] = int(gaze_entry.get())
        custom_settings["blink_rate_threshold"] = int(blink_entry.get())
        
        # Save settings
        save_settings(custom_settings)
        # Update thresholds
        update_thresholds()
        # Close window
        settings_window.destroy()

    settings_window = tk.Tk()
    settings_window.title("Drowsiness Detection Settings")
    settings_window.geometry("400x600")

    # Create settings form
    ttk.Label(settings_window, text="Detection Thresholds").pack(pady=5)
    
    ttk.Label(settings_window, text="Eye AR Threshold:").pack()
    ear_entry = ttk.Entry(settings_window)
    ear_entry.insert(0, str(custom_settings["eye_ar_threshold"]))
    ear_entry.pack()
    
    ttk.Label(settings_window, text="Consecutive Frames:").pack()
    ear_frames_entry = ttk.Entry(settings_window)
    ear_frames_entry.insert(0, str(custom_settings["eye_ar_consecutive_frames"]))
    ear_frames_entry.pack()
    
    ttk.Label(settings_window, text="Yawn Threshold:").pack()
    yawn_entry = ttk.Entry(settings_window)
    yawn_entry.insert(0, str(custom_settings["yawn_threshold"]))
    yawn_entry.pack()
    
    ttk.Label(settings_window, text="Horizontal Tilt:").pack()
    h_tilt_entry = ttk.Entry(settings_window)
    h_tilt_entry.insert(0, str(custom_settings["horizontal_tilt_threshold"]))
    h_tilt_entry.pack()
    
    ttk.Label(settings_window, text="Vertical Tilt:").pack()
    v_tilt_entry = ttk.Entry(settings_window)
    v_tilt_entry.insert(0, str(custom_settings["vertical_tilt_threshold"]))
    v_tilt_entry.pack()
    
    ttk.Label(settings_window, text="Gaze Threshold:").pack()
    gaze_entry = ttk.Entry(settings_window)
    gaze_entry.insert(0, str(custom_settings["gaze_threshold"]))
    gaze_entry.pack()
    
    ttk.Label(settings_window, text="Blink Rate (per min):").pack()
    blink_entry = ttk.Entry(settings_window)
    blink_entry.insert(0, str(custom_settings["blink_rate_threshold"]))
    blink_entry.pack()
    
    ttk.Button(settings_window, text="Save Settings", command=save_and_close).pack(pady=10)
    
    settings_window.mainloop()

def display_metrics(frame, metrics):
    """Display performance metrics on the frame with customizable settings"""
    if metrics:
        # Calculate session time
        current_time = time.time() - metrics["start_time"]
        hours = current_time / 3600
        minutes = (hours % 1) * 60
        seconds = (minutes % 1) * 60
        
        # Display session time
        cv2.putText(frame, f"Session Time: {int(hours)}h {int(minutes)}m {int(seconds):.0f}s", 
                    (10, 150), cv2.FONT_HERSHEY_DUPLEX, custom_settings["font_size"], 
                    custom_settings["text_color"], 2)
        
        # Display event counts
        cv2.putText(frame, f"Drowsiness Events: {metrics['drowsiness_events']}", 
                    (10, 180), cv2.FONT_HERSHEY_DUPLEX, custom_settings["font_size"], 
                    custom_settings["text_color"], 2)
        cv2.putText(frame, f"Yawns: {metrics['yawn_count']}", 
                    (10, 210), cv2.FONT_HERSHEY_DUPLEX, custom_settings["font_size"], 
                    custom_settings["text_color"], 2)
        cv2.putText(frame, f"Gaze Away: {metrics['gaze_away_count']}", 
                    (10, 240), cv2.FONT_HERSHEY_DUPLEX, custom_settings["font_size"], 
                    custom_settings["text_color"], 2)
        cv2.putText(frame, f"Head Tilts: {metrics['head_tilt_events']}", 
                    (10, 270), cv2.FONT_HERSHEY_DUPLEX, custom_settings["font_size"], 
                    custom_settings["text_color"], 2)
        
        # Display FPS
        fps = metrics.get('fps', 0)
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                    (10, 300), cv2.FONT_HERSHEY_DUPLEX, custom_settings["font_size"], 
                    custom_settings["text_color"], 2)

# Load settings at startup
custom_settings = load_settings()
update_thresholds()  # Apply loaded settings

# Initialize variables
EYE_AR_THRESH = custom_settings["eye_ar_threshold"]
EYE_AR_CONSEC_FRAMES = custom_settings["eye_ar_consecutive_frames"]
YAWN_THRESH = custom_settings["yawn_threshold"]
HORIZONTAL_TILT_THRESH = custom_settings["horizontal_tilt_threshold"]
VERTICAL_TILT_THRESH = custom_settings["vertical_tilt_threshold"]
GAZE_THRESH = custom_settings["gaze_threshold"]

# Alert severity levels and colors
ALERT_LEVELS = {
    "low": (0, 255, 0),    # Green
    "medium": (0, 255, 255), # Yellow
    "high": (0, 0, 255)    # Red
}

# Function to play alarm sound with different sounds for different alerts
def sound_alarm(path, alert_type="default"):
    global alarm_status
    global alarm_status2
    global alarm_status3

    pygame.mixer.init()
    
    # Different alarm sounds for different alerts
    alarm_sounds = {
        "drowsiness": path,
        "yawn": path.replace("C:\\Users\\nagav\\Downloads\\mixkit-alert-alarm-1005.wav", "_yawn.mp3"),
        "head": path.replace("C:\\Users\\nagav\\Downloads\\mixkit-alert-alarm-1005.wav", "_head.mp3"),
        "gaze": path.replace("C:\\Users\\nagav\\Downloads\\mixkit-alert-alarm-1005.wav", "_gaze.mp3"),
        "default": "C:\\Users\\nagav\\Downloads\\mixkit-alert-alarm-1005.wav"
    }
    
    sound_path = alarm_sounds.get(alert_type, alarm_sounds["default"])
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play(-1)

    while alarm_status or alarm_status2 or alarm_status3:
        continue

    pygame.mixer.music.stop()

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "7669187283:AAHhwo8EABp-NbD2qL0NxPuGx5vqKPFBdw4"
TELEGRAM_CHAT_ID = "1224043946"

def send_telegram_alert(message, image=None):
    """Send alert message to Telegram"""
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        
        # Send message
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        
        # Send image if provided
        if image is not None:
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', image)
            photo = io.BytesIO(buffer)
            photo.name = 'alert.jpg'
            
            # Send photo
            bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=photo)
            
    except Exception as e:
        print(f"Error sending Telegram alert: {str(e)}")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect gaze direction
def get_gaze_direction(shape):
    # Get eye landmarks
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    
    # Calculate eye centers
    left_center = np.mean(left_eye, axis=0)
    right_center = np.mean(right_eye, axis=0)
    
    # Calculate gaze vector
    gaze_vector = right_center - left_center
    gaze_angle = math.degrees(math.atan2(gaze_vector[1], gaze_vector[0]))
    
    # Normalize angle to be between -90 and 90
    if gaze_angle > 90:
        gaze_angle = gaze_angle - 180
    elif gaze_angle < -90:
        gaze_angle = gaze_angle + 180
    
    return gaze_angle

# Function to calculate head pose with improved accuracy
def get_head_position(shape):
    # Get facial landmarks for head pose estimation
    nose_tip = shape[30]
    left_eye_left = shape[36]
    left_eye_right = shape[39]
    right_eye_left = shape[42]
    right_eye_right = shape[45]
    nose_bridge = shape[27]
    chin = shape[8]
    
    # Calculate horizontal head tilt (left-right)
    left_eye_center = np.mean([left_eye_left, left_eye_right], axis=0)
    right_eye_center = np.mean([right_eye_left, right_eye_right], axis=0)
    
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    
    # Calculate horizontal tilt angle
    horizontal_tilt = math.degrees(math.atan2(dy, dx))
    
    # Normalize horizontal tilt to be 0 when face is straight
    horizontal_tilt = horizontal_tilt if horizontal_tilt <= 90 else horizontal_tilt - 180
    
    # Calculate vertical tilt (up-down)
    nose_to_chin_dx = chin[0] - nose_bridge[0]
    nose_to_chin_dy = chin[1] - nose_bridge[1]
    vertical_tilt = math.degrees(math.atan2(nose_to_chin_dy, nose_to_chin_dx)) - 90
    
    return horizontal_tilt, vertical_tilt

# Function to calculate the final EAR
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

# Function to calculate mouth opening distance
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Function to display performance metrics
def display_metrics(frame, metrics):
    """Display performance metrics on the frame with customizable settings"""
    if metrics:
        # Calculate session time
        current_time = time.time() - metrics["start_time"]
        hours = current_time / 3600
        minutes = (hours % 1) * 60
        seconds = (minutes % 1) * 60
        
        # Display session time
        cv2.putText(frame, f"Session Time: {int(hours)}h {int(minutes)}m {int(seconds):.0f}s", 
                    (10, 150), cv2.FONT_HERSHEY_DUPLEX, custom_settings["font_size"], 
                    custom_settings["text_color"], 2)
        
        # Display event counts
        cv2.putText(frame, f"Drowsiness Events: {metrics['drowsiness_events']}", 
                    (10, 180), cv2.FONT_HERSHEY_DUPLEX, custom_settings["font_size"], 
                    custom_settings["text_color"], 2)
        cv2.putText(frame, f"Yawns: {metrics['yawn_count']}", 
                    (10, 210), cv2.FONT_HERSHEY_DUPLEX, custom_settings["font_size"], 
                    custom_settings["text_color"], 2)
        cv2.putText(frame, f"Gaze Away: {metrics['gaze_away_count']}", 
                    (10, 240), cv2.FONT_HERSHEY_DUPLEX, custom_settings["font_size"], 
                    custom_settings["text_color"], 2)
        cv2.putText(frame, f"Head Tilts: {metrics['head_tilt_events']}", 
                    (10, 270), cv2.FONT_HERSHEY_DUPLEX, custom_settings["font_size"], 
                    custom_settings["text_color"], 2)
        
        # Display FPS
        fps = metrics.get('fps', 0)
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                    (10, 300), cv2.FONT_HERSHEY_DUPLEX, custom_settings["font_size"], 
                    custom_settings["text_color"], 2)

# Function to log session data
def log_session_data(metrics):
    """Log session data to CSV and generate report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = "session_logs.csv"
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Duration(s)", "Drowsiness Events", 
                            "Yawn Count", "Gaze Away Count", "Head Tilt Events", 
                            "FPS", "CPU Usage", "Memory Usage", "Frames Processed"])
    
    # Append session data
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            metrics["duration"],
            metrics["drowsiness_events"],
            metrics["yawn_count"],
            metrics["gaze_away_count"],
            metrics["head_tilt_events"],
            metrics["fps"],
            psutil.cpu_percent(),
            psutil.virtual_memory().percent,
            metrics["frame_count"]
        ])
    
    # Generate detailed report
    report_data = {
        "time": metrics["data"]["time"],
        "ear": metrics["data"]["ear"],
        "yawn": metrics["data"]["yawn"],
        "head_tilt": metrics["data"]["head_tilt"],
        "duration": metrics["duration"],
        "start_time": metrics["start_time"],
        "drowsiness_events": metrics["drowsiness_events"],
        "yawn_count": metrics["yawn_count"],
        "head_tilt_events": metrics["head_tilt_events"],
        "fps": metrics["fps"],
        "frame_count": metrics["frame_count"]
    }
    
    generate_session_report(report_data)

# Function to generate detailed session report
def generate_session_report(metrics, output_dir="reports"):
    """Generate a detailed session report with statistics and visualizations"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate statistics
    total_time = metrics["duration"]
    hours = total_time / 3600
    
    # Create report data
    report_data = {
        "Session Summary": {
            "Start Time": datetime.fromtimestamp(metrics["start_time"]).strftime("%Y-%m-%d %H:%M:%S"),
            "End Time": datetime.fromtimestamp(metrics["start_time"] + metrics["duration"]).strftime("%Y-%m-%d %H:%M:%S"),
            "Total Duration": f"{int(hours)}h {int((hours % 1) * 60)}m",
            "Average FPS": f"{metrics['fps']:.1f}"
        },
        "Drowsiness Statistics": {
            "Total Events": metrics["drowsiness_events"],
            "Events per Hour": f"{metrics['drowsiness_events'] / hours:.1f}",
            "Average Duration": f"{metrics['drowsiness_events'] * 30 / 60:.1f}s"
        },
        "Yawning Statistics": {
            "Total Yawns": metrics["yawn_count"],
            "Yawns per Hour": f"{metrics['yawn_count'] / hours:.1f}",
            "Average Duration": f"{metrics['yawn_count'] * 3:.1f}s"
        },
        "Head Movement": {
            "Total Tilts": metrics["head_tilt_events"],
            "Tilts per Hour": f"{metrics['head_tilt_events'] / hours:.1f}",
            "Horizontal Threshold": f"{10}°",
            "Vertical Threshold": f"{10}°"
        },
        "System Performance": {
            "Average CPU Usage": f"{psutil.cpu_percent():.1f}%",
            "Average Memory Usage": f"{psutil.virtual_memory().percent:.1f}%",
            "Frames Processed": metrics['frame_count']
        }
    }
    
    # Create HTML report
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }
            .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
            .metric { background: white; padding: 10px; border-radius: 3px; }
            .chart { margin: 20px 0; }
            h2 { color: #333; }
        </style>
    </head>
    <body>
        <h1>Drowsiness Detection Session Report</h1>
    """
    
    # Add session summary
    html_content += "<div class='section'>\n"
    html_content += "<h2>Session Summary</h2>\n"
    html_content += "<div class='metrics'>\n"
    for key, value in report_data["Session Summary"].items():
        html_content += f"<div class='metric'><strong>{key}:</strong> {value}</div>\n"
    html_content += "</div></div>\n"
    
    # Add detailed statistics
    for section, stats in report_data.items():
        if section == "Session Summary":
            continue
        html_content += f"<div class='section'>\n"
        html_content += f"<h2>{section}</h2>\n"
        html_content += "<div class='metrics'>\n"
        for key, value in stats.items():
            html_content += f"<div class='metric'><strong>{key}:</strong> {value}</div>\n"
        html_content += "</div></div>\n"
    
    # Add charts
    html_content += "<div class='section chart'>\n"
    html_content += "<h2>Event Timeline</h2>\n"
    html_content += "<img src='event_timeline.png' alt='Event Timeline'>\n"
    html_content += "</div>\n"
    
    html_content += "</body>\n</html>\n"
    
    # Save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"report_{timestamp}.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    # Generate event timeline chart
    plt.figure(figsize=(10, 4))
    plt.plot(metrics["time"], metrics["ear"], label="EAR", color="#00ff00")
    plt.plot(metrics["time"], metrics["yawn"], label="Yawn", color="#ff0000")
    plt.plot(metrics["time"], metrics["head_tilt"], label="Head Tilt", color="#00ffff")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("Event Timeline")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"event_timeline_{timestamp}.png"))
    plt.close()
    
    print(f"Session report generated at: {report_path}")
    return report_path

# Function to display system status
def display_system_status(frame, metrics):
    # Calculate system resource usage
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    # Update FPS calculation
    metrics["frame_count"] += 1
    if time.time() - metrics["last_fps_update"] >= 1.0:
        metrics["fps"] = metrics["frame_count"]
        metrics["frame_count"] = 0
        metrics["last_fps_update"] = time.time()
    
    # Display system status
    cv2.putText(frame, f"FPS: {metrics['fps']}", (10, 330), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"CPU: {cpu_percent}%", (10, 360), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Memory: {memory.percent}%", (10, 390), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
    
    # Display pause status
    if paused:
        cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 50, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

# User profile management
class UserProfile:
    def __init__(self, username):
        self.username = username
        self.profile_file = f"profiles/{username}.json"
        self.load_profile()

    def load_profile(self):
        if not os.path.exists("profiles"):
            os.makedirs("profiles")
        
        if os.path.exists(self.profile_file):
            with open(self.profile_file, 'r') as f:
                data = json.load(f)
                self.settings = data.get('settings', self.get_default_settings())
                self.stats = data.get('stats', self.get_default_stats())
        else:
            self.settings = self.get_default_settings()
            self.stats = self.get_default_stats()
            self.save_profile()

    def get_default_settings(self):
        return {
            "eye_threshold": 0.3,
            "yawn_threshold": 20,
            "gaze_threshold": 30,
            "head_tilt_threshold": 10,
            "alert_email": "",
            "alert_phone": "",
            "notification_enabled": False,
            "custom_alarm_sound": ""
        }

    def get_default_stats(self):
        return {
            "total_sessions": 0,
            "total_duration": 0,
            "drowsiness_events": 0,
            "yawn_count": 0,
            "gaze_away_count": 0,
            "head_tilt_events": 0
        }

    def save_profile(self):
        with open(self.profile_file, 'w') as f:
            json.dump({
                'settings': self.settings,
                'stats': self.stats
            }, f, indent=4)

    def update_stats(self, session_data):
        self.stats["total_sessions"] += 1
        self.stats["total_duration"] += session_data["duration"]
        self.stats["drowsiness_events"] += session_data["drowsiness_events"]
        self.stats["yawn_count"] += session_data["yawn_count"]
        self.stats["gaze_away_count"] += session_data["gaze_away_count"]
        self.stats["head_tilt_events"] += session_data["head_tilt_events"]
        self.save_profile()

# Email notification system
class NotificationSystem:
    def init(self, email, password):
        self.email = email
        self.password = password
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587

    def send_alert(self, subject, message):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email
            msg['To'] = self.email
            msg['Subject'] = subject

            msg.attach(MIMEText(message, 'plain'))

            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email, self.password)
            server.send_message(msg)
            server.quit()
            return True
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            return False

# Data visualization
class DataVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(6, 4), facecolor='black')
        self.canvas = FigureCanvas(self.fig)
        self.data = {
            'time': [],
            'ear': [],
            'yawn': [],
            'gaze': [],
            'head_tilt': []
        }
        self.colors = {
            'ear': '#00ff00',      # Green
            'yawn': '#ff0000',     # Red
            'gaze': '#ffff00',     # Yellow
            'head_tilt': '#00ffff' # Cyan
        }
        self.max_data_points = 100  # Keep only last 100 points

    def update_data(self, metrics):
        current_time = time.time() - metrics["start_time"]
        self.data['time'].append(current_time)
        self.data['ear'].append(metrics.get('current_ear', 0))
        self.data['yawn'].append(metrics.get('current_yawn', 0))
        self.data['gaze'].append(metrics.get('current_gaze', 0))
        self.data['head_tilt'].append(metrics.get('current_head_tilt', 0))

        # Keep only last max_data_points values
        if len(self.data['time']) > self.max_data_points:
            for key in self.data:
                self.data[key] = self.data[key][-self.max_data_points:]

    def draw_graph(self, frame, x_offset=0, y_offset=0, width=None, height=None):
        try:
            # Clear the figure
            self.fig.clear()

            # Create a subplot with a dark background
            ax = self.fig.add_subplot(111)
            ax.set_facecolor('#000000')
            self.fig.patch.set_facecolor('#000000')

            # Plot each metric if there is enough data
            if len(self.data['time']) > 1:
                times = np.array(self.data['time']) - self.data['time'][0]  # Relative time

                # Plot EAR (Eye Aspect Ratio)
                ax.plot(times, self.data['ear'], color=self.colors['ear'], 
                        label='EAR', linewidth=2, alpha=0.8)
                
                # Plot Yawn
                ax.plot(times, self.data['yawn'], color=self.colors['yawn'], 
                        label='Yawn', linewidth=2, alpha=0.8)
                
                # Plot Gaze
                ax.plot(times, self.data['gaze'], color=self.colors['gaze'], 
                        label='Gaze', linewidth=2, alpha=0.8)
                
                # Plot Head Tilt
                ax.plot(times, self.data['head_tilt'], color=self.colors['head_tilt'], 
                        label='Head Tilt', linewidth=2, alpha=0.8)

                ax.grid(True, color='#333333', alpha=0.3)
                ax.set_xlabel('Time (s)', color='white')
                ax.set_ylabel('Value', color='white')
                ax.tick_params(colors='white')
                
                # Improve legend positioning
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                         facecolor='#333333', edgecolor='#666666', labelcolor='white')

            self.fig.tight_layout()
            self.canvas.draw()
            graph_image = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8)
            graph_image = graph_image.reshape(self.canvas.get_width_height()[::-1] + (4,))
            graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2BGR)

            if width is None:
                width = frame.shape[1] // 3
            if height is None:
                height = frame.shape[0] // 3

            graph_image = cv2.resize(graph_image, (width, height))
            
            # Position graph in bottom-right corner
            x_offset = frame.shape[1] - width - 10
            y_offset = frame.shape[0] - height - 10
            roi = frame[y_offset:y_offset+height, x_offset:x_offset+width]

            if roi.shape == graph_image.shape:
                frame[y_offset:y_offset+height, x_offset:x_offset+width] = graph_image

        except Exception as e:
            print(f"Error drawing graph: {str(e)}")

# Settings GUI
class SettingsGUI:
    def init(self, user_profile):
        self.root = tk.Tk()
        self.root.title("Drowsiness Detection Settings")
        self.user_profile = user_profile
        self.setup_gui()

    def setup_gui(self):
        # Create settings frame
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding="10")
        settings_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        # Threshold settings
        ttk.Label(settings_frame, text="Eye Threshold:").grid(row=0, column=0, sticky="w")
        self.eye_threshold = ttk.Entry(settings_frame)
        self.eye_threshold.insert(0, str(self.user_profile.settings["eye_threshold"]))
        self.eye_threshold.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(settings_frame, text="Yawn Threshold:").grid(row=1, column=0, sticky="w")
        self.yawn_threshold = ttk.Entry(settings_frame)
        self.yawn_threshold.insert(0, str(self.user_profile.settings["yawn_threshold"]))
        self.yawn_threshold.grid(row=1, column=1, padx=5, pady=2)

        # Notification settings
        ttk.Label(settings_frame, text="Alert Email:").grid(row=2, column=0, sticky="w")
        self.alert_email = ttk.Entry(settings_frame)
        self.alert_email.insert(0, self.user_profile.settings["alert_email"])
        self.alert_email.grid(row=2, column=1, padx=5, pady=2)

        # Save button
        ttk.Button(settings_frame, text="Save Settings", command=self.save_settings).grid(row=3, column=0, columnspan=2, pady=10)

    def save_settings(self):
        self.user_profile.settings["eye_threshold"] = float(self.eye_threshold.get())
        self.user_profile.settings["yawn_threshold"] = float(self.yawn_threshold.get())
        self.user_profile.settings["alert_email"] = self.alert_email.get()
        self.user_profile.save_profile()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# Check and download required files
print("Checking for required files...")
check_and_download_files()

# Argument parser setup
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default=r'C:\\Users\\nagav\\Downloads\\loud-emergency-alarm-54635.mp3', help="path to alarm sound file")
args = vars(ap.parse_args())

# Constants
EYE_AR_THRESH = custom_settings["eye_ar_threshold"]
EYE_AR_CONSEC_FRAMES = custom_settings["eye_ar_consecutive_frames"]
YAWN_THRESH = custom_settings["yawn_threshold"]
HORIZONTAL_TILT_THRESH = custom_settings["horizontal_tilt_threshold"]
VERTICAL_TILT_THRESH = custom_settings["vertical_tilt_threshold"]
GAZE_THRESH = custom_settings["gaze_threshold"]

# Load face detector and predictor
print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Start video stream
print("-> Starting Video Stream")
vs = cv2.VideoCapture(args["webcam"])
vs.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
vs.set(cv2.CAP_PROP_FPS, 30)
time.sleep(1.0)  # Give the camera time to warm up

# Initialize features
current_user = UserProfile("default")
visualizer = DataVisualizer()

# Main loop
while True:
    if not paused:
        ret, frame = vs.read()
        if not ret or frame is None:
            print("Error: Could not read frame from video stream")
            break
            
        # Convert frame to RGB if it's not already
        if len(frame.shape) == 2:  # If grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:  # If RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            
            try:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Get head position and gaze direction
                horizontal_tilt, vertical_tilt = get_head_position(shape)
                gaze_angle = get_gaze_direction(shape)

                eye = final_ear(shape)
                ear = eye[0]
                leftEye = eye[1]
                rightEye = eye[2]

                distance = lip_distance(shape)

                # Draw facial landmarks with confidence indicators
                confidence = 1.0 - (abs(ear - EYE_AR_THRESH) / EYE_AR_THRESH)
                color = ALERT_LEVELS["high"] if confidence < 0.5 else ALERT_LEVELS["medium"] if confidence < 0.8 else ALERT_LEVELS["low"]
                
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, color, 1)
                cv2.drawContours(frame, [rightEyeHull], -1, color, 1)

                lip = shape[48:60]
                cv2.drawContours(frame, [lip], -1, color, 1)

                # Gaze Direction Detection
                if abs(gaze_angle) > GAZE_THRESH:
                    cv2.putText(frame, "GAZE ALERT!", (10, 270), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f"Gaze Angle: {gaze_angle:.1f}°", (10, 300), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                    update_metrics(performance_metrics, ear, distance, abs(horizontal_tilt), "gaze_away")

                # Head Position Detection
                head_alert = False
                head_message = ""
                
                # Check vertical tilt (nodding)
                if vertical_tilt < -VERTICAL_TILT_THRESH:
                    head_alert = True
                    head_message = "Head Down!"
                    update_metrics(performance_metrics, ear, distance, abs(horizontal_tilt), "head_tilt")
                elif vertical_tilt > VERTICAL_TILT_THRESH:
                    head_alert = True
                    head_message = "Head Up!"
                    update_metrics(performance_metrics, ear, distance, abs(horizontal_tilt), "head_tilt")
                    
                # Check horizontal tilt (left-right)
                if abs(horizontal_tilt) > HORIZONTAL_TILT_THRESH:
                    head_alert = True
                    head_message = "Head Tilted!" if not head_message else head_message + " Head Tilted!"
                    update_metrics(performance_metrics, ear, distance, abs(horizontal_tilt), "head_tilt")
                
                if head_alert:
                    if not alarm_status3:
                        alarm_status3 = True
                        if args["alarm"] != "":
                            t = Thread(target=sound_alarm, args=(args["alarm"], "head"))
                            t.daemon = True
                            t.start()
                    cv2.putText(frame, f"HEAD ALERT: {head_message}", (10, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                else:
                    alarm_status3 = False

                # Drowsiness Detection (Eye Closure)
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not alarm_status:
                            alarm_status = True
                            if args["alarm"] != "":
                                t = Thread(target=sound_alarm, args=(args["alarm"], "drowsiness"))
                                t.daemon = True
                                t.start()
                            message = f"🚨 DROWSINESS ALERT 🚨\n\n" \
                             f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" \
                             f"Event: Drowsiness Detected\n" \
                             f"Eye Aspect Ratio: {ear:.2f}"
                            send_telegram_alert(message, frame)
                            update_metrics(performance_metrics, ear, distance, abs(horizontal_tilt), "drowsiness")

                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    alarm_status = False

                # Yawning Detection
                if distance > YAWN_THRESH:
                    cv2.putText(frame, "Yawn Alert", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
                    if not alarm_status2 and not saying:
                        alarm_status2 = True
                        if args["alarm"] != "":
                            t = Thread(target=sound_alarm, args=(args["alarm"], "yawn"))
                            t.daemon = True
                            t.start()
                        message = f"⚠️ YAWN ALERT ⚠️\n\n" \
                         f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" \
                         f"Event: Yawn Detected\n" \
                         f"Mouth Distance: {distance:.2f}"
                        send_telegram_alert(message, frame)
                        update_metrics(performance_metrics, ear, distance, abs(horizontal_tilt), "yawn")
                else:
                    alarm_status2 = False

                # Display confidence score
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (300, 150), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

            except Exception as e:
                print(f"Error processing frame: {str(e)}")

        # Update visualizer
        visualizer.update_data({
            "start_time": performance_metrics["start_time"],
            "current_ear": ear,
            "current_yawn": distance,
            "current_gaze": abs(gaze_angle),
            "current_head_tilt": abs(horizontal_tilt)
        })
        visualizer.draw_graph(frame)

        # Display system status
        display_system_status(frame, performance_metrics)
        
        # Display performance metrics
        display_metrics(frame, performance_metrics)

    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF

    # Stop Alarm when 's' is pressed
    if key == ord("s"):
        alarm_status = False
        alarm_status2 = False
        alarm_status3 = False
        pygame.mixer.music.stop()

    # Pause/Resume when 'p' is pressed
    if key == ord("p"):
        paused = not paused
        if paused:
            cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 50, 30), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)

    # Save session data when 'l' is pressed
    if key == ord("l"):
        log_session_data(performance_metrics)
        cv2.putText(frame, "Detailed Report Generated", (frame.shape[1]//2 - 120, 60), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    # Show settings menu when 't' is pressed
    if key == ord("t"):
        threading.Thread(target=display_settings_menu).start()

    # Save user stats when quitting
    if key == ord("q"):
        session_data = {
            "duration": time.time() - performance_metrics["start_time"],
            "drowsiness_events": performance_metrics["drowsiness_events"],
            "yawn_count": performance_metrics["yawn_count"],
            "gaze_away_count": performance_metrics["gaze_away_count"],
            "head_tilt_events": performance_metrics["head_tilt_events"]
        }
        current_user.update_stats(session_data)
        log_session_data(performance_metrics)
        break

# Cleanup
cv2.destroyAllWindows()
vs.release()