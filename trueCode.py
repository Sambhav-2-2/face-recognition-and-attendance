from pathlib import Path
from datetime import datetime, date, time as dt_time
import pandas as pd  # pip install pandas
import cv2
import torch
import numpy as np
import threading
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Shared state
latest_frame = None
frame_lock = threading.Lock()
new_frame_event = threading.Event()
stop_event = threading.Event()

results_lock = threading.Lock()
last_results = []  # list of (box, name, score)

CAM_INDEX = "http://10.204.157.178:5000/video_feed"  # change if needed
SIM_THRESHOLD = 0.5  # similarity threshold for recognition
DWELL_TIME = 3.0  # seconds a face must be detected before logging attendance

# ---- Night Alert Settings (9 PM to 6 AM) ----
ENABLE_NIGHT_ALERTS = True  # Set to False to disable
NIGHT_START_HOUR = 21  # 9 PM (24-hour format)
NIGHT_END_HOUR = 6     # 6 AM (24-hour format)
ALERT_COOLDOWN = 300   # seconds between alerts for same person (5 minutes)

# Email settings (Gmail example - enable "App Passwords" in Google Account)
EMAIL_ALERTS = True  # Set to False to disable email
SMTP_SERVER = "your mail"
SMTP_PORT = 123
SENDER_EMAIL = "sender email"  # CHANGE THIS
SENDER_PASSWORD = "your pw"   # CHANGE THIS - use App Password, not regular password
ALERT_EMAIL = "alert mails"  # CHANGE THIS - where to send alerts

# SMS settings (using Twilio - sign up at twilio.com for free trial)
SMS_ALERTS = True  # Set to True to enable SMS (requires Twilio setup)
TWILIO_ACCOUNT_SID = "your sid"  # CHANGE THIS
TWILIO_AUTH_TOKEN = "auth_token"    # CHANGE THIS
TWILIO_PHONE_NUMBER ="your phone num"      # CHANGE THIS - your Twilio number
ALERT_PHONE_NUMBER = "reciever phone num"       # CHANGE THIS - where to send SMS

# ---- Attendance logging (dedup per day) ----
ATT_DIR = Path("attendance_logs")
USE_EXCEL = True  # set False to use CSV instead of Excel

attendance_lock = threading.Lock()
attendance_date = date.today()
attendance_today = set()  # names already logged today

# ---- Face dwell time tracking (anti-spoofing) ----
dwell_tracker = {}  # {name: first_seen_time}
dwell_lock = threading.Lock()

# ---- Night alert tracking ----
alert_tracker = {}  # {name: last_alert_time}
alert_lock = threading.Lock()
ALERT_IMAGES_DIR = Path("alert_images")
ALERT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

def _attendance_file_for(d: date) -> Path:
    ATT_DIR.mkdir(parents=True, exist_ok=True)
    return ATT_DIR / (f"attendance_{d.isoformat()}.xlsx" if USE_EXCEL
                      else f"attendance_{d.isoformat()}.csv")

def _load_existing_names(filepath: Path) -> set:
    try:
        if not filepath.exists():
            return set()
        if USE_EXCEL:
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
        if "Name" in df.columns:
            return set(map(str, df["Name"].tolist()))
        return set()
    except Exception:
        # If file is corrupted or empty, start fresh for dedup purposes
        return set()

def init_attendance():
    """Initialize today's in-memory dedup set from existing file (if any)."""
    global attendance_date, attendance_today
    attendance_date = date.today()
    fpath = _attendance_file_for(attendance_date)
    attendance_today = _load_existing_names(fpath)

def is_nighttime() -> bool:
    """Check if current time is during alert hours (9 PM to 6 AM)."""
    current_hour = datetime.now().hour
    if NIGHT_START_HOUR > NIGHT_END_HOUR:  # crosses midnight
        return current_hour >= NIGHT_START_HOUR or current_hour < NIGHT_END_HOUR
    else:
        return NIGHT_START_HOUR <= current_hour < NIGHT_END_HOUR

def send_email_alert(name: str, similarity: float, image_path: Path):
    """Send email alert with face detection details and image."""
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ALERT_EMAIL
        msg['Subject'] = f"🚨 SECURITY ALERT: Face Detected - {name}"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "KNOWN PERSON" if name != "Unknown" else "UNKNOWN PERSON"
        
        body = f"""
SECURITY ALERT - Face Detection During Night Hours

Time: {timestamp}
Status: {status}
Identity: {name}
Confidence: {similarity:.2%}

Location: Camera {CAM_INDEX}
Alert Reason: Face detected outside authorized hours (9 PM - 6 AM)

{"⚠️ This is an UNKNOWN person - immediate action may be required!" if name == "Unknown" else "This person is in the system but detected during restricted hours."}

Please review the attached image and take appropriate action if necessary.

---
Automated Face Recognition Security System
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach image if available
        if image_path.exists():
            with open(image_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=image_path.name)
                msg.attach(img)
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        
        print(f"[ALERT] Email sent to {ALERT_EMAIL}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to send email alert: {repr(e)}")
        return False

def send_sms_alert(name: str, similarity: float):
    """Send SMS alert via Twilio."""
    try:
        from twilio.rest import Client
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "KNOWN" if name != "Unknown" else "UNKNOWN"
        
        message_body = (
            f"🚨 SECURITY ALERT\n"
            f"Face Detected: {status}\n"
            f"Name: {name}\n"
            f"Time: {timestamp}\n"
            f"Camera: {CAM_INDEX}\n"
            f"Confidence: {similarity:.0%}"
        )
        
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=ALERT_PHONE_NUMBER
        )
        
        print(f"[ALERT] SMS sent to {ALERT_PHONE_NUMBER}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to send SMS alert: {repr(e)}")
        return False

def check_and_send_night_alert(name: str, similarity: float, frame: np.ndarray):
    """
    Check if it's nighttime and send alert if a face is detected.
    Respects cooldown period to avoid spam.
    """
    if not ENABLE_NIGHT_ALERTS or not is_nighttime():
        return
    
    current_time = time.time()
    
    with alert_lock:
        # Check cooldown
        if name in alert_tracker:
            time_since_last = current_time - alert_tracker[name]
            if time_since_last < ALERT_COOLDOWN:
                return  # Still in cooldown
        
        # Update tracker
        alert_tracker[name] = current_time
    
    # Save alert image
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"alert_{timestamp_str}_{name.replace(' ', '_')}.jpg"
    image_path = ALERT_IMAGES_DIR / image_filename
    cv2.imwrite(str(image_path), frame)
    
    print(f"[ALERT] 🚨 NIGHTTIME DETECTION: {name} (similarity: {similarity:.2f})")
    
    # Send alerts
    if EMAIL_ALERTS:
        send_email_alert(name, similarity, image_path)
    
    if SMS_ALERTS:
        send_sms_alert(name, similarity)

def _roll_date_if_needed():
    """If date changed at midnight, rotate dedup set."""
    global attendance_date, attendance_today
    today = date.today()
    if today != attendance_date:
        attendance_date = today
        attendance_today = _load_existing_names(_attendance_file_for(today))

def mark_attendance(name: str, similarity: float) -> bool:
    """
    Log 'name' once per day with timestamp + similarity.
    Only logs after the face has been continuously detected for DWELL_TIME seconds.
    Returns True if written this call, False if it was already present today or still dwelling.
    """
    if not name or name == "Unknown":
        return False

    current_time = time.time()
    
    # Check dwell time requirement
    with dwell_lock:
        if name not in dwell_tracker:
            # First time seeing this face
            dwell_tracker[name] = current_time
            print(f"[DWELL] {name} detected, tracking for {DWELL_TIME}s...")
            return False
        
        elapsed = current_time - dwell_tracker[name]
        if elapsed < DWELL_TIME:
            # Still dwelling, not ready to log
            remaining = DWELL_TIME - elapsed
            if int(elapsed) != int(elapsed - 1):  # Print every second
                print(f"[DWELL] {name}: {elapsed:.1f}s / {DWELL_TIME}s")
            return False
        
        # Dwell time met, proceed to log
        # Remove from tracker so we don't keep checking
        del dwell_tracker[name]

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with attendance_lock:
        _roll_date_if_needed()
        if name in attendance_today:
            return False  # already logged today

        fpath = _attendance_file_for(attendance_date)
        record = {"Timestamp": ts, "Name": name, "Similarity": round(float(similarity), 4)}

        try:
            if USE_EXCEL:
                # Append using openpyxl if file exists to avoid rewriting entire file
                if fpath.exists():
                    from openpyxl import load_workbook
                    wb = load_workbook(fpath)
                    ws = wb.active
                    ws.append([record["Timestamp"], record["Name"], record["Similarity"]])
                    wb.save(fpath)
                else:
                    pd.DataFrame([record]).to_excel(fpath, index=False)
            else:
                # CSV: append without header if file exists
                pd.DataFrame([record]).to_csv(
                    fpath, mode="a", index=False, header=not fpath.exists()
                )
        except Exception as e:
            print(f"[ERROR] Attendance write error: {repr(e)}")
            return False

        attendance_today.add(name)
        print(f"[ATTENDANCE] ✓ {name} logged @ {ts} (sim={record['Similarity']}) after {DWELL_TIME}s verification")
        return True


def capture_loop():
    """Continuously capture frames and keep the latest frame only."""
    global latest_frame
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Failed to open camera")
        stop_event.set()
        return

    print("[INFO] Camera capture started")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed")
            break
        with frame_lock:
            latest_frame = frame.copy()
        # Notify processor that a new frame is available
        new_frame_event.set()
        # Small sleep to avoid hammering the CPU
        time.sleep(0.03)  # ~30 FPS capture rate

    cap.release()
    print("[INFO] Camera capture stopped")


def processing_loop():
    """Process frames one-at-a-time. Initialize models here to keep CUDA context local."""
    global latest_frame, last_results

    # Initialize device, models and DB inside this thread (important for CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Processor initializing on {device}...")
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    client = chromadb.PersistentClient(path="./face_db")
    collection = client.get_or_create_collection(
        name="students",
        metadata={"hnsw:space": "cosine"}
    )

    # Load database once at startup for efficiency
    db_cache = {"embeddings": [], "names": [], "count": 0}
    
    def refresh_db_cache():
        results_db = collection.get(include=["embeddings", "metadatas"])
        db_cache["count"] = len(results_db["embeddings"])
        if db_cache["count"] > 0:
            db_cache["embeddings"] = np.array(results_db["embeddings"])
            db_cache["names"] = [m["name"] for m in results_db["metadatas"]]
        print(f"[INFO] Loaded {db_cache['count']} faces from database")
    
    refresh_db_cache()
    print("[INFO] Processor ready")

    while not stop_event.is_set():
        # Wait until a new frame has been signalled
        new_frame_event.wait(timeout=1.0)
        
        # Grab the latest frame
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
            new_frame_event.clear()

        if frame is None:
            continue

        try:
            # Convert to PIL for MTCNN
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Detect faces and get aligned face tensors
            boxes, _ = mtcnn.detect(img)
            faces = mtcnn(img)

            results = []
            if boxes is not None and faces is not None:
                for face_tensor, box in zip(faces, boxes):
                    if face_tensor is None:
                        continue

                    # Get embedding
                    with torch.no_grad():
                        emb_t = model(face_tensor.unsqueeze(0).to(device))
                    embedding = emb_t.cpu().numpy().flatten()

                    # Search using cached database
                    if db_cache["count"] == 0:
                        name, score = "Unknown", 0.0
                    else:
                        sims = cosine_similarity(
                            embedding.reshape(1, -1), 
                            db_cache["embeddings"]
                        )[0]
                        best_idx = int(np.argmax(sims))
                        name = db_cache["names"][best_idx]
                        score = float(sims[best_idx])
                        
                        if score < SIM_THRESHOLD:
                            name = "Unknown"

                    results.append((box, name, score))
                    
                    # Check for nighttime alerts (send immediately, no dwell time needed)
                    if ENABLE_NIGHT_ALERTS and is_nighttime():
                        check_and_send_night_alert(name, score, frame)
                    
                    # Mark attendance for recognized faces (requires dwell time during day)
                    if name != "Unknown" and score >= SIM_THRESHOLD and not is_nighttime():
                        mark_attendance(name, score)
            
            # Clean up dwell tracker for faces no longer detected
            with dwell_lock:
                detected_names = {name for _, name, _ in results if name != "Unknown"}
                # Remove faces that disappeared before completing dwell time
                for tracked_name in list(dwell_tracker.keys()):
                    if tracked_name not in detected_names:
                        elapsed = time.time() - dwell_tracker[tracked_name]
                        if elapsed < DWELL_TIME:
                            print(f"[DWELL] {tracked_name} lost after {elapsed:.1f}s (needed {DWELL_TIME}s)")
                        del dwell_tracker[tracked_name]

            # Publish results atomically
            with results_lock:
                last_results = results

        except Exception as e:
            print(f"[ERROR] Processing exception: {repr(e)}")


def main():
    init_attendance()
    print("[INFO] Starting face recognition attendance system...")
    print(f"[INFO] Similarity threshold: {SIM_THRESHOLD}")
    print(f"[INFO] Dwell time requirement: {DWELL_TIME} seconds (anti-spoofing)")
    print(f"[INFO] Faces must be detected continuously for {DWELL_TIME}s before logging")
    
    if ENABLE_NIGHT_ALERTS:
        print(f"[INFO] 🌙 Night alerts ENABLED ({NIGHT_START_HOUR}:00 - {NIGHT_END_HOUR}:00)")
        print(f"[INFO] Alert cooldown: {ALERT_COOLDOWN}s between alerts per person")
        if EMAIL_ALERTS:
            print(f"[INFO] 📧 Email alerts enabled → {ALERT_EMAIL}")
        if SMS_ALERTS:
            print(f"[INFO] 📱 SMS alerts enabled → {ALERT_PHONE_NUMBER}")
        if is_nighttime():
            print("[WARNING] ⚠️ CURRENTLY IN NIGHT MODE - Security alerts active!")
    else:
        print("[INFO] Night alerts disabled")
    
    # Start threads
    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    proc_thread = threading.Thread(target=processing_loop, daemon=True)
    cap_thread.start()
    proc_thread.start()

    # Main: display loop (must run on main thread for cv2.imshow)
    print("[INFO] Press 'q' to quit")
    while not stop_event.is_set():
        with frame_lock:
            display_frame = latest_frame.copy() if latest_frame is not None else None

        if display_frame is None:
            # Nothing captured yet: show a black frame
            h, w = 480, 640
            display_frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(display_frame, "Waiting for camera...", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show night mode indicator
        if ENABLE_NIGHT_ALERTS and is_nighttime():
            cv2.putText(display_frame, "NIGHT MODE - ALERTS ACTIVE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw last results
        with results_lock:
            results_copy = list(last_results)

        for box, name, score in results_copy:
            x1, y1, x2, y2 = map(int, box.tolist())
            
            # Color coding based on status and time
            if ENABLE_NIGHT_ALERTS and is_nighttime():
                color = (0, 0, 255)  # Red during night mode (alert!)
            elif name == "Unknown":
                color = (0, 165, 255)  # Orange for unknown
            else:
                color = (0, 255, 0)  # Green for known
            
            # Check if person is currently dwelling (being verified)
            is_dwelling = False
            dwell_progress = 0.0
            with dwell_lock:
                if name in dwell_tracker and name != "Unknown":
                    is_dwelling = True
                    elapsed = time.time() - dwell_tracker[name]
                    dwell_progress = min(elapsed / DWELL_TIME, 1.0)
                    if not is_nighttime():  # Only show yellow during day
                        color = (0, 255, 255)  # Yellow during dwell time
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Show dwell progress if applicable (not during night mode)
            if is_dwelling and not is_nighttime():
                label = f"{name} - Verifying... {dwell_progress*100:.0f}%"
            elif is_nighttime():
                label = f"{name} - NIGHT ALERT!"
            else:
                label = f"{name} ({score:.2f})"
            
            cv2.putText(display_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Live Face Recognition", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

        time.sleep(0.01)

    # Clean up
    print("[INFO] Shutting down...")
    cv2.destroyAllWindows()
    cap_thread.join(timeout=2.0)
    proc_thread.join(timeout=2.0)
    print("[INFO] System stopped")


if __name__ == "__main__":
    main()