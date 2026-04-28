import cv2
import numpy as np
import mediapipe as mp
import os
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ==========================================
# 1. ตั้งค่าและโหลดโมเดล (ทำแค่ครั้งเดียวประหยัดเวลา!)
# ==========================================
MODEL_DIR = "../models/saved_models/v5_expert_tuning" 
MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_model_v5.keras")
ACTIONS_PATH = os.path.join(MODEL_DIR, "actions.npy")

print("=========================================")
print("Sign Language Interpreter V5")
print("=========================================")
print("⏳ กำลังโหลดสมองกล AI กรุณารอสักครู่...")
model = load_model(MODEL_PATH)
actions = np.load(ACTIONS_PATH)
print(f"✅ โหลดสำเร็จ! พร้อมจำแนก {len(actions)} คำศัพท์\n")

# เตรียมระบบหน้าต่างเลือกไฟล์
root = tk.Tk()
root.withdraw() 
root.attributes('-topmost', True) 

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
# 3. Main Loop: ระบบวิเคราะห์วิดีโอต่อเนื่อง
# ==========================================
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True: # ลูปหลักของโปรแกรม (รันไปเรื่อยๆ จนกว่าจะสั่งปิด)
        
        # 3.1 เด้งหน้าต่างให้เลือกไฟล์
        root.update() # รีเฟรชหน้าต่างให้พร้อมทำงาน
        TEST_VIDEO_PATH = filedialog.askopenfilename(
            title="เลือกวิดีโอภาษามือที่ต้องการทดสอบ",
            initialdir=os.path.abspath("../data/videos"),
            filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
        )

        # ตรวจสอบการยกเลิกเลือกไฟล์
        if not TEST_VIDEO_PATH:
            print("👋 ยกเลิกการเลือกไฟล์ ปิดโปรแกรมเรียบร้อยครับ")
            break # ออกจากลูปหลัก และปิดโปรแกรม

        print(f"\n🎬 กำลังวิเคราะห์วิดีโอ: {os.path.basename(TEST_VIDEO_PATH)}")
        
        # 3.2 รีเซ็ตตัวแปรสำหรับคลิปใหม่
        sequence = []
        prediction_history = [] 
        threshold = 0.50 
        current_action = "Analyzing..."
        last_frame = None 

        cap = cv2.VideoCapture(TEST_VIDEO_PATH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 30 
        
        skip_video = False # เอาไว้เช็คว่าผู้ใช้กดข้ามวิดีโอหรือไม่

        # 3.3 ลูปเล่นวิดีโอ
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            if width > 1280:
                frame = cv2.resize(frame, (int(width*0.5), int(height*0.5)))
                
            last_frame = frame.copy() 

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            draw_styled_landmarks(frame, results)
            
            if results.pose_landmarks:
                raw_keypoints = extract_and_normalize_keypoints_optimized(results)
                engineered_features = extract_joint_angles(raw_keypoints)
                
                sequence.append(engineered_features)
                sequence = sequence[-60:] 
                
                pad_length = 60 - len(sequence)
                padded_sequence = sequence + [np.zeros(392)] * pad_length
                
                res = model.predict(np.expand_dims(padded_sequence, axis=0), verbose=0)[0]
                best_match_idx = np.argmax(res)
                confidence = res[best_match_idx]
                
                if confidence > threshold:
                    current_action = actions[best_match_idx]
                    prediction_history.append((current_action, confidence * 100))
                
                cv2.rectangle(frame, (0,0), (640, 50), (245, 117, 16), -1)
                cv2.putText(frame, f'{current_action} ({confidence*100:.1f}%)', (15, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Video Sign Language Interpreter', frame)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'): # ถ้ากด q ระหว่างเล่นคลิป ให้ข้ามไปโชว์สรุปผลเลย
                skip_video = True
                break
            elif key == ord(' '): 
                cv2.waitKey(-1)

        cap.release()
       # ==========================================
        # 4. วาดหน้าต่างสรุปผลตอนจบ (Summary Overlay)
        # ==========================================
        if last_frame is not None:
            # 🌟 ทริคแก้ปัญหา: บังคับปรับขนาดเฟรมสุดท้ายเป็นหน้าจอ HD (1280x720) 
            # เพื่อให้มีพื้นที่กว้างและสูงพอสำหรับเขียนข้อความเสมอ ไม่ทับกันแน่นอน
            last_frame_resized = cv2.resize(last_frame, (1280, 720))
            h, w = last_frame_resized.shape[:2]
            
            overlay = last_frame_resized.copy()
            # วาดกล่องดำเว้นขอบ
            cv2.rectangle(overlay, (40, 40), (w-40, h-40), (20, 20, 20), -1)
            summary_frame = cv2.addWeighted(overlay, 0.85, last_frame_resized, 0.15, 0)
            
            cv2.putText(summary_frame, "--- PREDICTION SUMMARY ---", (80, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 215, 255), 3, cv2.LINE_AA)
            
            if not prediction_history:
                cv2.putText(summary_frame, "No confident predictions found.", (80, 180), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                summary_dict = {}
                for action, conf in prediction_history:
                    if action not in summary_dict:
                        summary_dict[action] = []
                    summary_dict[action].append(conf)
                
                y_offset = 180
                cv2.putText(summary_frame, "DETECTED WORD", (80, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(summary_frame, "MAX CONF", (450, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(summary_frame, "AVG CONF", (700, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(summary_frame, "FRAMES", (950, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                y_offset += 30
                cv2.line(summary_frame, (80, y_offset), (1150, y_offset), (255, 255, 255), 2)
                y_offset += 50
                
                sorted_actions = sorted(summary_dict.items(), key=lambda item: len(item[1]), reverse=True)
                
                # แสดงแค่ Top 5 จะได้ไม่ล้นหน้าจอ
                for action, conf_list in sorted_actions[:5]:
                    max_conf = max(conf_list)
                    avg_conf = sum(conf_list) / len(conf_list)
                    frame_count = len(conf_list)
                    
                    cv2.putText(summary_frame, f"- {action}", (80, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(summary_frame, f"{max_conf:.1f}%", (450, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.putText(summary_frame, f"{avg_conf:.1f}%", (700, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                    cv2.putText(summary_frame, f"{frame_count}", (950, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                    
                    y_offset += 50

            # คำแนะนำในการไปต่อ (ยึดตามความสูง h=720 จึงไม่ไปทับด้านบนแน่นอน)
            cv2.putText(summary_frame, "Press 'N' to select New Video", (80, h-90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(summary_frame, "Press 'Q' to Quit Program", (80, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # ขยายหน้าต่างให้พอดีกับรูปที่ขยายแล้ว
            cv2.namedWindow('Video Sign Language Interpreter ', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Video Sign Language Interpreter ', 1280, 720)
            cv2.imshow('Video Sign Language Interpreter ', summary_frame)
            
            user_choice = None
            while True: 
                key = cv2.waitKey(0) & 0xFF
                if key == ord('n') or key == ord('N'):
                    user_choice = 'new'
                    break
                elif key == ord('q') or key == ord('Q'):
                    user_choice = 'quit'
                    break
            
            if user_choice == 'quit':
                print("👋 ปิดโปรแกรมเรียบร้อยครับ")
                break
            elif user_choice == 'new':
                print("-" * 50)
                continue

cv2.destroyAllWindows()