import cv2
import numpy as np
import mediapipe as mp
import os
import time
from tensorflow.keras.models import load_model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==========================================
# 1. ตั้งค่าโฟลเดอร์และโหลดโมเดล
# ==========================================
MODEL_DIR = "../models/saved_models/v5_expert_tuning" 
MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_model_v5.keras")
ACTIONS_PATH = os.path.join(MODEL_DIR, "actions.npy")

print("=========================================")
print("🤖 Sign Language Live Studio")
print("=========================================")
print("⏳ กำลังโหลดโมเดลและคำศัพท์...")
model = load_model(MODEL_PATH)
actions = np.load(ACTIONS_PATH)
print(f"✅ โหลดสำเร็จ! พร้อมจำแนก {len(actions)} คำศัพท์\n")

# ==========================================
# 2. ฟังก์ชัน MediaPipe และ Feature Engineering
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 
    270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 
    312, 311, 310, 415
]

def extract_and_normalize_keypoints_optimized(results):
    ref_x, ref_y = 0.0, 0.0
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        ref_x, ref_y = nose.x, nose.y

    def process_landmarks(landmarks, expected_length, has_visibility=False):
        if landmarks:
            if has_visibility:
                return np.array([[p.x - ref_x, p.y - ref_y, p.z, p.visibility] for p in landmarks.landmark]).flatten()
            else:
                return np.array([[p.x - ref_x, p.y - ref_y, p.z] for p in landmarks.landmark]).flatten()
        else:
            return np.zeros(expected_length)

    def process_selected_face_landmarks(landmarks, indices):
        if landmarks:
            selected_points = []
            for idx in indices:
                p = landmarks.landmark[idx]
                selected_points.extend([p.x - ref_x, p.y - ref_y, p.z])
            return np.array(selected_points)
        else:
            return np.zeros(len(indices) * 3)

    pose = process_landmarks(results.pose_landmarks, 132, has_visibility=True)
    lh = process_landmarks(results.left_hand_landmarks, 63, has_visibility=False)
    rh = process_landmarks(results.right_hand_landmarks, 63, has_visibility=False)
    face_lips = process_selected_face_landmarks(results.face_landmarks, LIP_INDICES)

    return np.concatenate([pose, lh, rh, face_lips])

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def extract_joint_angles(frame_data):
    pose = frame_data[:132].reshape(33, 4)[:, :3]
    lh = frame_data[132:195].reshape(21, 3)
    rh = frame_data[195:258].reshape(21, 3)
    
    angles = []
    if np.any(pose):
        angles.extend([calculate_angle(pose[11], pose[13], pose[15]), calculate_angle(pose[12], pose[14], pose[16]),
                       calculate_angle(pose[23], pose[11], pose[13]), calculate_angle(pose[24], pose[12], pose[14])])
    else: angles.extend([0]*4)
        
    if np.any(lh):
        angles.extend([calculate_angle(lh[0], lh[1], lh[4]), calculate_angle(lh[0], lh[5], lh[8]),
                       calculate_angle(lh[0], lh[9], lh[12]), calculate_angle(lh[0], lh[13], lh[16]),
                       calculate_angle(lh[0], lh[17], lh[20])])
    else: angles.extend([0]*5)
        
    if np.any(rh):
        angles.extend([calculate_angle(rh[0], rh[1], rh[4]), calculate_angle(rh[0], rh[5], rh[8]),
                       calculate_angle(rh[0], rh[9], rh[12]), calculate_angle(rh[0], rh[13], rh[16]),
                       calculate_angle(rh[0], rh[17], rh[20])])
    else: angles.extend([0]*5)
        
    return np.concatenate([frame_data, np.array(angles) / 180.0])

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 

# ==========================================
# 3. ระบบจัดการปุ่มและเมาส์คลิกบนจอ (Mouse Callback)
# ==========================================
# สร้างตัวแปร Global เพื่อส่งสัญญาณว่าโดนคลิกแล้ว
switch_cam_flag = False 

def check_button_click(event, x, y, flags, param):
    global switch_cam_flag
    # ดักจับอีเวนต์เมื่อกดคลิกเมาส์ซ้าย (Left Button Down)
    if event == cv2.EVENT_LBUTTONDOWN:
        # เช็คพิกัดของปุ่ม (คำนวณจากหน้าจอทั้งหมดที่กว้าง 1280 + 450 = 1730px)
        # กล้องกว้าง 1280px แถบด้านขวากว้าง 450px ปุ่มเราวาดไว้ตรง X: 1300-1710, Y: 630-690
        if 1300 <= x <= 1710 and 630 <= y <= 690:
            switch_cam_flag = True

# ==========================================
# 4. ระบบ Real-time Detection + Live Dashboard
# ==========================================
sequence = []     
predictions_buffer = []  
threshold = 0.70  
current_action = "Waiting..."
display_confidence = 0.0
session_stats = {} 

current_cam_idx = 0
max_cameras = 5

print(f"🎥 กำลังเปิดกล้อง (Camera {5})...")
cap = cv2.VideoCapture(5)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# สร้างหน้าต่างและผูกระบบเมาส์เข้ากับหน้าต่างนี้
WINDOW_NAME = 'Sign Language Studio'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, check_button_click) # ผูกฟังก์ชันคลิกเมาส์เข้ากับหน้าต่าง

with mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=0 
) as holistic:
    print("✨ เริ่มทำภาษามือหน้ากล้องได้เลย! เอาเมาส์คลิกปุ่ม SWITCH CAMERA ได้เลย")
    
    prev_time = 0 
    frame_counter = 0 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1280, 720))

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        draw_styled_landmarks(frame, results)
        
        # --- 4.1 ระบบ AI ทายผล ---
        if results.pose_landmarks: 
            raw_keypoints = extract_and_normalize_keypoints_optimized(results)
            engineered_features = extract_joint_angles(raw_keypoints) 
            
            sequence.append(engineered_features)
            sequence = sequence[-60:] 
            
            if len(sequence) == 60 and frame_counter % 2 == 0:
                input_data = np.expand_dims(sequence, axis=0)
                res = model(input_data, training=False).numpy()[0]
                best_match_idx = np.argmax(res)
                confidence = res[best_match_idx]
                predicted_word = actions[best_match_idx]
                
                predictions_buffer.append(predicted_word)
                predictions_buffer = predictions_buffer[-15:]
                
                if confidence > threshold and predictions_buffer.count(predicted_word) >= 10:
                    current_action = predicted_word
                    display_confidence = confidence * 100
                    
                    if current_action not in session_stats:
                        session_stats[current_action] = []
                    session_stats[current_action].append(display_confidence)

        # --- 4.2 สร้าง Live Dashboard ---
        sidebar = np.zeros((720, 450, 3), dtype=np.uint8)
        
        cv2.rectangle(sidebar, (0, 0), (450, 150), (30, 30, 30), -1)
        
        # แสดงสถานะกล้องปัจจุบันไว้ด้านบนสุด
        cv2.putText(sidebar, f"CAM: {current_cam_idx}", (350, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # เปลี่ยนสี FPS ตามความลื่นไหล (เขียว=ลื่น, ส้ม=พอใช้, แดง=หน่วง)
        fps_color = (0, 255, 0) if fps > 20 else ((0, 165, 255) if fps > 10 else (0, 0, 255))
        cv2.putText(sidebar, f"FPS: {int(fps)}", (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

        cv2.putText(sidebar, "LIVE DETECTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(sidebar, current_action, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        if current_action != "Waiting...":
            cv2.putText(sidebar, f"{display_confidence:.1f}%", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 255), 2)

        cv2.putText(sidebar, "SESSION STATISTICS", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y_offset = 240
        cv2.putText(sidebar, "WORD", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        cv2.putText(sidebar, "MAX", (200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        cv2.putText(sidebar, "AVG", (280, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        cv2.putText(sidebar, "FRM", (360, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
        cv2.line(sidebar, (20, y_offset + 10), (430, y_offset + 10), (100, 100, 100), 1)
        
        y_offset += 40
        
        if session_stats:
            sorted_stats = sorted(session_stats.items(), key=lambda item: len(item[1]), reverse=True)
            for action, conf_list in sorted_stats[:10]:
                max_conf = max(conf_list)
                avg_conf = sum(conf_list) / len(conf_list)
                frames = len(conf_list)
                
                color_word = (0, 255, 0) if action == current_action else (220, 220, 220)
                
                cv2.putText(sidebar, f"{action}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_word, 2)
                cv2.putText(sidebar, f"{max_conf:.0f}%", (200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
                cv2.putText(sidebar, f"{avg_conf:.0f}%", (280, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.putText(sidebar, f"{frames}", (360, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                y_offset += 35
        else:
            cv2.putText(sidebar, "No actions detected yet.", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        # 🌟 วาดปุ่ม "SWITCH CAMERA" ให้ดูคลิกได้ (อยู่มุมขวาล่างของแถบดำ)
        button_color = (60, 180, 60) # สีเขียว
        cv2.rectangle(sidebar, (20, 630), (430, 690), button_color, -1) # ตัวปุ่ม
        cv2.rectangle(sidebar, (20, 630), (430, 690), (255, 255, 255), 2) # ขอบปุ่มสีขาว
        
        # กะระยะให้ข้อความอยู่ตรงกลางปุ่มพอดี
        cv2.putText(sidebar, "SWITCH CAMERA", (110, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # รวมร่างกล้อง + Dashboard
        final_dashboard = np.hstack((frame, sidebar))
        cv2.imshow(WINDOW_NAME, final_dashboard)

        # --- 4.3 ระบบควบคุมคีย์บอร์ด & จัดการปุ่มกด ---
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
            
        # เช็คว่าผู้ใช้กดปุ่ม 'c' บนคีย์บอร์ด หรือ เอาเมาส์ไปคลิกปุ่มบนจอ
        if switch_cam_flag or key == ord('c') or key == ord('C'):
            print(f"🔄 ได้รับคำสั่งสลับกล้อง...")
            cap.release() 
            current_cam_idx = (current_cam_idx + 1) % max_cameras
            
            cap = cv2.VideoCapture(current_cam_idx)
            
            if not cap.isOpened():
                print(f"⚠️ ไม่พบกล้อง Index {current_cam_idx}, กลับไปใช้กล้อง 0")
                current_cam_idx = 0
                cap = cv2.VideoCapture(current_cam_idx)
                
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            sequence = []     
            predictions_buffer = []  
            current_action = "Camera Switched..."
            display_confidence = 0.0
            switch_cam_flag = False # รีเซ็ตสัญญาณปุ่มกด
            print(f"✅ เปิดกล้อง {current_cam_idx} สำเร็จ!")

    cap.release()
    cv2.destroyAllWindows()