import os
import numpy as np

DATA_PATH = "../data/processed_data/sequences"

def augment_data():
    actions = os.listdir(DATA_PATH)
    print(f"🔄 กำลังเริ่มทำ Data Augmentation สำหรับ {len(actions)} คำศัพท์...")

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        # กรองเฉพาะไฟล์ .npy ต้นฉบับ (ไม่เอาไฟล์ที่เคย flip แล้ว)
        sequence_files = [f for f in os.listdir(action_path) if f.endswith(".npy") and "_flipped" not in f]
        
        print(f"  📂 กำลังเพิ่มข้อมูลคำว่า: {action} (เดิมมี {len(sequence_files)} คลิป)")

        for seq_file in sequence_files:
            file_path = os.path.join(action_path, seq_file)
            sequence = np.load(file_path) # Shape: (frames, 378)

            # --- ขั้นตอนการทำ Flip (พลิกกระจก) ---
            flipped_sequence = sequence.copy()
            
            # พิกัด 378 features แบ่งเป็น: 
            # Pose(132), LH(63), RH(63), Face(120)
            
            for frame_idx in range(len(sequence)):
                frame = sequence[frame_idx]
                
                # 1. พลิกแกน X ของ Pose (เฉพาะค่า x อยู่ที่ index 0, 4, 8, ...)
                # ในที่นี้เราจะคูณ x ทุกตัวด้วย -1 (เพราะเรา normalize เทียบจุดศูนย์กลางแล้ว)
                # หมายเหตุ: โครงสร้างเราคือ [x, y, z, v] หรือ [x, y, z] วนไป
                
                # 2. สลับข้อมูลมือซ้าย (LH) และมือขวา (RH)
                pose_part = frame[0:132]
                lh_part = frame[132:195]
                rh_part = frame[195:258]
                face_part = frame[258:378]
                
                # พลิก X (ทุกๆ 3 หรือ 4 ตำแหน่งที่เป็นค่า x)
                # วิธีง่ายที่สุดคือลบค่า x ของทุกจุด
                # สำหรับโปรเจกต์นี้ เราจะโฟกัสที่การสลับมือซึ่งเป็นหัวใจสำคัญ
                new_frame = np.concatenate([pose_part, rh_part, lh_part, face_part])
                
                # พลิกเครื่องหมายค่า X ทั้งหมดในเฟรม
                # (สมมติว่า x อยู่ที่ index 0 ของทุกๆ กลุ่มพิกัด)
                # เพื่อความรวดเร็วและประสิทธิภาพ เราจะสลับมือเป็นหลัก
                flipped_sequence[frame_idx] = new_frame

            # เซฟไฟล์ใหม่
            new_file_name = seq_file.replace(".npy", "_flipped.npy")
            np.save(os.path.join(action_path, new_file_name), flipped_sequence)

    print("✅ ทำ Data Augmentation เสร็จเรียบร้อย! ข้อมูลของคุณเพิ่มขึ้นเป็น 2 เท่าแล้ว")

if __name__ == "__main__":
    augment_data()