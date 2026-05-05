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

# 2. ฟังก์ชันสกัดและ Normalize Keypoints 
def extract_and_normalize_keypoints_optimized(results):
    ref_x, ref_y, ref_z = 0.0, 0.0, 0.0
    shoulder_width = 1.0 # ค่าตั้งต้น
    
    if results.pose_landmarks:
        nose = results.pose_landmarks.landmark[0]
        ref_x, ref_y, ref_z = nose.x, nose.y, nose.z
        
        l_shoulder = results.pose_landmarks.landmark[11]
        r_shoulder = results.pose_landmarks.landmark[12]
        shoulder_width = ((l_shoulder.x - r_shoulder.x)**2 + (l_shoulder.y - r_shoulder.y)**2)**0.5
        shoulder_width = max(shoulder_width, 0.01)

    def process_landmarks(landmarks, expected_length, has_visibility=False):
        if landmarks:
            if has_visibility:
                # ⭐️ แก้ไข: ลบ .flatten() ออก ให้ return เป็น (N, 4)
                return np.array([[(p.x - ref_x)/shoulder_width, 
                                  (p.y - ref_y)/shoulder_width, 
                                  (p.z - ref_z)/shoulder_width, 
                                  p.visibility] for p in landmarks.landmark])
            else:
                # ⭐️ แก้ไข: ลบ .flatten() ออก ให้ return เป็น (N, 3)
                return np.array([[(p.x - ref_x)/shoulder_width, 
                                  (p.y - ref_y)/shoulder_width, 
                                  (p.z - ref_z)/shoulder_width] for p in landmarks.landmark])
        else:
            # ⭐️ แก้ไข: คืนค่าเป็น (N, 3) หรือ (N, 4) ถ้าไม่มีข้อมูล
            cols = 4 if has_visibility else 3
            # expected_length ของคุณคือจำนวนจุดรวม เช่น Pose คือ 132 -> จุดคือ 33
            num_points = expected_length // cols 
            return np.zeros((num_points, cols))

    # สกัดข้อมูล (จะได้เป็น 2D array: (33,4), (21,3), (21,3))
    pose = process_landmarks(results.pose_landmarks, 132, has_visibility=True)
    lh = process_landmarks(results.left_hand_landmarks, 63, has_visibility=False)
    rh = process_landmarks(results.right_hand_landmarks, 63, has_visibility=False)

    return pose, lh, rh # ⭐️ คืนค่าแบบยังไม่ประกอบร่าง

# 3. ฟังก์ชันสำหรับหมุนภาพ (Data Augmentation - Rotation)
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# 4. ฟังก์ชันสำหรับหมุนพิกัด 3D (Data Augmentation - Rotation)
def rotate_3d_y_axis(landmarks_array, angle_degrees):
    """ 
    หมุนพิกัด 3D รอบแกน Y (จำลองกล้องย้ายไปซ้าย-ขวา) 
    รับค่า input เป็น array ของพิกัดขนาด (N, 3) หรือ (N, 4)
    """
    angle_rad = np.radians(angle_degrees)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    
    # สร้างเมทริกซ์การหมุน (Rotation Matrix รอบแกน Y)
    rotation_matrix = np.array([
        [cos_theta, 0, sin_theta],
        [0, 1, 0],
        [-sin_theta, 0, cos_theta]
    ])
    
    # ก๊อปปี้ข้อมูลเพื่อไม่ให้ทับของเดิม
    rotated_landmarks = landmarks_array.copy()
    
    # ดึงเฉพาะพิกัด X, Y, Z มาหมุน (ถ้ามีค่า Visibility ต่อท้ายจะได้ไม่พัง)
    xyz = rotated_landmarks[:, :3]
    
    # จับคูณเมทริกซ์
    rotated_xyz = np.dot(xyz, rotation_matrix.T)
    
    # เอาค่าที่หมุนแล้วใส่กลับเข้าไป
    rotated_landmarks[:, :3] = rotated_xyz
    
    return rotated_landmarks

def process_and_augment(pose, lh, rh, angle_degrees=0):
    """
    ฟังก์ชันหมุน 3D (ถ้า angle != 0) และประกอบร่าง + ทุบแบนให้เสร็จ
    """
    if angle_degrees != 0:
        angle_rad = np.radians(angle_degrees)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        
        rot_mat = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])
        
        pose_rot = pose.copy()
        pose_rot[:, :3] = np.dot(pose[:, :3], rot_mat.T)
        
        lh_rot = lh.copy()
        lh_rot[:, :3] = np.dot(lh[:, :3], rot_mat.T)
        
        rh_rot = rh.copy()
        rh_rot[:, :3] = np.dot(rh[:, :3], rot_mat.T)
    else:
        pose_rot = pose
        lh_rot = lh
        rh_rot = rh

    return np.concatenate([pose_rot.flatten(), lh_rot.flatten(), rh_rot.flatten()])
# 5. ฟังก์ชันหลักสำหรับประมวลผลวิดีโอ
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

                seq_orig, seq_flip, seq_rot, seq_fast, seq_slow = [], [], [], [], []
                seq_rot_left, seq_rot_right = [], []

                frame_count = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1

                    # --- ก. ภาพต้นฉบับ ---
                    img_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res_orig = holistic.process(img_orig)

                    if res_orig.pose_landmarks:
                        # extract ครั้งเดียว ใช้ได้ทุก sequence
                        p, l, r = extract_and_normalize_keypoints_optimized(res_orig)
                        kp_flat = process_and_augment(p, l, r, 0)

                        # 1. ต้นฉบับ
                        seq_orig.append(kp_flat)

                        # 2. Fast (เก็บเฉพาะเฟรมคี่)
                        if frame_count % 2 != 0:
                            seq_fast.append(kp_flat)

                        # 3. Slow (เบิ้ลเฟรมคู่ ทำให้ยาวขึ้น 1.5 เท่า)
                        seq_slow.append(kp_flat)
                        if frame_count % 2 == 0:
                            seq_slow.append(kp_flat)

                        # 4. 3D Rotation ซ้าย 15 องศา
                        seq_rot_left.append(process_and_augment(
                            rotate_3d_y_axis(p, 15),
                            rotate_3d_y_axis(l, 15),
                            rotate_3d_y_axis(r, 15), 0))

                        # 5. 3D Rotation ขวา 15 องศา
                        seq_rot_right.append(process_and_augment(
                            rotate_3d_y_axis(p, -15),
                            rotate_3d_y_axis(l, -15),
                            rotate_3d_y_axis(r, -15), 0))

                    # --- ข. สลับซ้าย-ขวา (Mirror/Flip 2D) ---
                    img_flip = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                    res_flip = holistic.process(img_flip)
                    if res_flip.pose_landmarks:
                        p_flip, l_flip, r_flip = extract_and_normalize_keypoints_optimized(res_flip)
                        seq_flip.append(process_and_augment(p_flip, l_flip, r_flip, 0))

                    # --- ค. เอียงภาพ (Rotation 5 องศา 2D) ---
                    img_rot = cv2.cvtColor(rotate_image(frame, 5), cv2.COLOR_BGR2RGB)
                    res_rot = holistic.process(img_rot)
                    if res_rot.pose_landmarks:
                        p_rot, l_rot, r_rot = extract_and_normalize_keypoints_optimized(res_rot)
                        seq_rot.append(process_and_augment(p_rot, l_rot, r_rot, 0))

                cap.release()

                # บันทึกไฟล์ทั้ง 7 รูปแบบ
                if len(seq_orig) > 0:
                    np.save(save_path_orig, np.array(seq_orig))
                if len(seq_flip) > 0:
                    np.save(os.path.join(word_dir, f"{video_id}_flip.npy"), np.array(seq_flip))
                if len(seq_rot) > 0:
                    np.save(os.path.join(word_dir, f"{video_id}_rot.npy"), np.array(seq_rot))
                if len(seq_fast) > 0:
                    np.save(os.path.join(word_dir, f"{video_id}_fast.npy"), np.array(seq_fast))
                if len(seq_slow) > 0:
                    np.save(os.path.join(word_dir, f"{video_id}_slow.npy"), np.array(seq_slow))
                if len(seq_rot_left) > 0:
                    np.save(os.path.join(word_dir, f"{video_id}_rot3Dl.npy"), np.array(seq_rot_left))
                if len(seq_rot_right) > 0:
                    np.save(os.path.join(word_dir, f"{video_id}_rot3Dr.npy"), np.array(seq_rot_right))

                print(f"  ✅ {video_id} สกัดสำเร็จ (รวม Augment ได้ 7 ไฟล์)")

            # --- แจ้งเตือนความคืบหน้าเข้า LINE ---
            progress_percent = ((idx + 1) / total_words) * 100
            if progress_percent >= next_notify_threshold:
                message = (f"Bot] สกัด Keypoints คืบหน้า: {int(progress_percent)}%\n"
                           f"📂 สำเร็จแล้ว {idx + 1}/{total_words} คำศัพท์\n"
                           f"📝 คำล่าสุด: {gloss}")
                send_line_notification(message)
                next_notify_threshold += 20
                
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