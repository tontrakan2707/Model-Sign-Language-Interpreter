import cv2
import numpy as np
import mediapipe as mp
import json
import os

# ตั้งค่าโฟลเดอร์ (แยกโฟลเดอร์ข้อสอบออกมาต่างหาก)
JSON_PATH = "../data/processed_data/filtered_top50_videos.json" # ไฟล์ 50 คำ
VIDEO_DIR = "../data/unseen_videos" # 🌟 โฟลเดอร์ที่เก็บคลิปวิดีโอข้อสอบ
OUTPUT_DIR = "../data/processed_data/test_sequences" # 🌟 โฟลเดอร์เก็บ Keypoints ข้อสอบ

mp_holistic = mp.solutions.holistic
LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]

# ฟังก์ชันสกัด Keypoints ตัวเดียวกับที่คุณใช้เป๊ะๆ
def extract_and_normalize_keypoints_optimized(results):
    ref_x, ref_y = 0.0, 0.0
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        ref_x, ref_y = nose.x, nose.y

    def process_landmarks(landmarks, expected_length, has_visibility=False):
        if landmarks:
            if has_visibility: return np.array([[p.x - ref_x, p.y - ref_y, p.z, p.visibility] for p in landmarks.landmark]).flatten()
            else: return np.array([[p.x - ref_x, p.y - ref_y, p.z] for p in landmarks.landmark]).flatten()
        else: return np.zeros(expected_length)

    def process_selected_face_landmarks(landmarks, indices):
        if landmarks:
            selected_points = []
            for idx in indices:
                p = landmarks.landmark[idx]
                selected_points.extend([p.x - ref_x, p.y - ref_y, p.z])
            return np.array(selected_points)
        else: return np.zeros(len(indices) * 3)

    pose = process_landmarks(results.pose_landmarks, 132, has_visibility=True)
    lh = process_landmarks(results.left_hand_landmarks, 63, has_visibility=False)
    rh = process_landmarks(results.right_hand_landmarks, 63, has_visibility=False)
    face_lips = process_selected_face_landmarks(results.face_landmarks, LIP_INDICES)
    return np.concatenate([pose, lh, rh, face_lips])

def extract_test_videos():
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ ไม่พบโฟลเดอร์วิดีโอข้อสอบ: {VIDEO_DIR}")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    print(f"🚀 เริ่มสกัด Keypoints สำหรับชุดข้อสอบ (Test Set) ...")
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for entry in data:
            gloss = entry['gloss']
            instances = entry['instances']
            word_dir = os.path.join(OUTPUT_DIR, gloss)
            os.makedirs(word_dir, exist_ok=True)
            
            for inst in instances:
                video_id = inst['video_id']
                video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
                
                # ถ้าไฟล์วิดีโอไม่อยู่ในโฟลเดอร์ unseen_videos ก็ข้ามไป
                if not os.path.exists(video_path):
                    continue
                    
                save_path = os.path.join(word_dir, f"{video_id}_test.npy")
                if os.path.exists(save_path):
                    continue

                cap = cv2.VideoCapture(video_path)
                seq_orig = []

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    img_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res_orig = holistic.process(img_orig)
                    if res_orig.pose_landmarks:
                        seq_orig.append(extract_and_normalize_keypoints_optimized(res_orig))
                cap.release()
                
                # 🌟 เซฟเฉพาะภาพต้นฉบับ ไม่ทำ Augmentation
                if len(seq_orig) > 0:
                    np.save(save_path, np.array(seq_orig))
                    print(f"  ✅ สกัดข้อสอบ {video_id}.mp4 ({gloss}) สำเร็จ!")

if __name__ == "__main__":
    extract_test_videos()
    print("🎉 สกัดข้อสอบเสร็จสมบูรณ์ พร้อมเอาไปทดสอบโมเดล!")