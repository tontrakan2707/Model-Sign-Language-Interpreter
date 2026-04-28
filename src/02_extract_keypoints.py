import cv2
import numpy as np
import mediapipe as mp
import json
import os
from notifier import send_line_notification

# ตั้งค่าสภาพแวดล้อมเพื่อลด log กวนใจ
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 1. กำหนด MediaPipe และ Index ของจุดริมฝีปาก (LIPS)
mp_holistic = mp.solutions.holistic
LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 
    270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 
    312, 311, 310, 415
]

# 2. ฟังก์ชันสกัดและ Normalize Keypoints (378 features)
def extract_and_normalize_keypoints_optimized(results):
    ref_x, ref_y = 0.0, 0.0
    if results.pose_landmarks:
        # ใช้จมูกเป็นจุดอ้างอิง (Reference point)
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

    # Pose: 33 points * 4 (x,y,z,v) = 132
    # Hands: 21 points * 3 (x,y,z) = 63 each
    # Lips: 40 points * 3 (x,y,z) = 120
    pose = process_landmarks(results.pose_landmarks, 132, has_visibility=True)
    lh = process_landmarks(results.left_hand_landmarks, 63, has_visibility=False)
    rh = process_landmarks(results.right_hand_landmarks, 63, has_visibility=False)
    face_lips = process_selected_face_landmarks(results.face_landmarks, LIP_INDICES)

    return np.concatenate([pose, lh, rh, face_lips])

# 3. ฟังก์ชันสำหรับหมุนภาพ (Data Augmentation - Rotation)
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# 4. ฟังก์ชันหลักสำหรับประมวลผลวิดีโอ
def process_videos(json_path, video_dir, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    total_words = len(data)
    next_notify_threshold = 10  # เริ่มแจ้งเตือนที่ 10%

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for idx, entry in enumerate(data):
            gloss = entry['gloss']
            instances = entry['instances']
            
            # สร้างโฟลเดอร์สำหรับคำศัพท์
            word_dir = os.path.join(output_dir, gloss)
            os.makedirs(word_dir, exist_ok=True)
            
            print(f"\n🎬 [{idx+1}/{total_words}] กำลังประมวลผลคำว่า: {gloss} ({len(instances)} คลิป)")
            
            for inst in instances:
                video_id = inst['video_id']
                video_path = os.path.join(video_dir, f"{video_id}.mp4")
                
                # Checkpoint: ถ้ามีไฟล์ต้นฉบับแล้วให้ข้าม (Skip)
                save_path_orig = os.path.join(word_dir, f"{video_id}_orig.npy")
                if os.path.exists(save_path_orig):
                    continue

                if not os.path.exists(video_path):
                    continue

                cap = cv2.VideoCapture(video_path)
                
                # เตรียม List เก็บข้อมูล 5 รูปแบบ (Augmentation)
                seq_orig, seq_flip, seq_rot, seq_fast, seq_slow = [], [], [], [], []
                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_count += 1
                    
                    # --- ก. ภาพต้นฉบับ (ใช้สำหรับ Fast และ Slow ด้วย) ---
                    img_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res_orig = holistic.process(img_orig)
                    if res_orig.pose_landmarks:
                        kp = extract_and_normalize_keypoints_optimized(res_orig)
                        
                        # 1. ต้นฉบับ
                        seq_orig.append(kp)
                        
                        # 2. แบบเร็ว (Fast - เก็บเฉพาะเฟรมคี่)
                        if frame_count % 2 != 0: seq_fast.append(kp)
                            
                        # 3. แบบช้า (Slow - เบิ้ลเฟรมคู่ ทำให้ยาวขึ้น 1.5 เท่า)
                        seq_slow.append(kp)
                        if frame_count % 2 == 0: seq_slow.append(kp)

                    # --- ข. สลับซ้าย-ขวา (Mirror/Flip) ---
                    img_flip = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                    res_flip = holistic.process(img_flip)
                    if res_flip.pose_landmarks:
                        seq_flip.append(extract_and_normalize_keypoints_optimized(res_flip))

                    # --- ค. เอียงภาพ (Rotation 5 องศา) ---
                    img_rot = cv2.cvtColor(rotate_image(frame, 5), cv2.COLOR_BGR2RGB)
                    res_rot = holistic.process(img_rot)
                    if res_rot.pose_landmarks:
                        seq_rot.append(extract_and_normalize_keypoints_optimized(res_rot))

                cap.release()
                
                # บันทึกไฟล์ทั้ง 5 รูปแบบ
                if len(seq_orig) > 0: np.save(save_path_orig, np.array(seq_orig))
                if len(seq_flip) > 0: np.save(os.path.join(word_dir, f"{video_id}_flip.npy"), np.array(seq_flip))
                if len(seq_rot) > 0: np.save(os.path.join(word_dir, f"{video_id}_rot.npy"), np.array(seq_rot))
                if len(seq_fast) > 0: np.save(os.path.join(word_dir, f"{video_id}_fast.npy"), np.array(seq_fast))
                if len(seq_slow) > 0: np.save(os.path.join(word_dir, f"{video_id}_slow.npy"), np.array(seq_slow))
                
                print(f"  ✅ {video_id} สกัดสำเร็จ (รวม Augment ได้ 5 ไฟล์)")

            # --- ส่วนการคำนวณและแจ้งเตือนความคืบหน้าเข้า LINE ---
            progress_percent = ((idx + 1) / total_words) * 100
            if progress_percent >= next_notify_threshold:
                message = (f"Bot] สกัด Keypoints คืบหน้า: {int(progress_percent)}%\n"
                           f"📂 สำเร็จแล้ว {idx + 1}/{total_words} คำศัพท์\n"
                           f"📝 คำล่าสุด: {gloss}")
                send_line_notification(message)
                next_notify_threshold += 10 # ขยับเป้าหมายครั้งถัดไป

if __name__ == "__main__":
    # ⚠️ อย่าลืมเช็คชื่อไฟล์ตรงนี้นะครับ (ควรเป็นไฟล์ที่คุณกรองมา 50 คำศัพท์)
    JSON_PATH = "../data/processed_data/filtered_top50_videos.json"
    VIDEO_DIR = "../data/videos"
    OUTPUT_DIR = "../data/processed_data/sequences"
    
    print("🚀 เริ่มต้นกระบวนการสกัด Keypoints + 5 Augmentations...")
    send_line_notification("Bot] เริ่มต้นงานสกัด Keypoints (50 คำ + 5 Augments) แล้วครับ!")
    
    try:
        process_videos(JSON_PATH, VIDEO_DIR, OUTPUT_DIR)
        print("🎉 สกัด Keypoints เสร็จสมบูรณ์ทั้งหมด!")
        send_line_notification("Bot] ภารกิจสำเร็จ 100%! สกัดและเพิ่มข้อมูลพร้อมเทรนแล้วครับ")
    except Exception as e:
        error_info = f"Bot] เกิดข้อผิดพลาด: {str(e)}"
        print(error_info)
        send_line_notification(error_info)