import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# 0. ฟังก์ชันคำนวณองศาข้อต่อ (อัปเกรดฟีเจอร์ให้ V5)
# ==========================================
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def add_angles_to_sequence(sequence_data):
    # เช็คก่อนว่าข้อมูลมี 378 ฟีเจอร์ใช่ไหม ถ้าเป็น 392 อยู่แล้วให้ข้ามไปเลย
    if sequence_data.shape[-1] == 392:
        return sequence_data
        
    new_sequence = []
    for frame_data in sequence_data:
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
            
        new_frame = np.concatenate([frame_data, np.array(angles) / 180.0])
        new_sequence.append(new_frame)
    return np.array(new_sequence)

# ==========================================
# 1. โหลดข้อมูลและโมเดล
# ==========================================
DATA_PATH = "../data/processed_data/sequences"
MODEL_DIR = "../models/saved_models/v5_expert_tuning"
MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_model_v5.keras")
ACTIONS_PATH = os.path.join(MODEL_DIR, "actions.npy")

# ⚠️ สำคัญมาก: โหลดรายชื่อคำศัพท์จาก actions.npy เพื่อให้ลำดับตรงกับสมองกลเป๊ะๆ 100%
actions = np.load(ACTIONS_PATH)
model = load_model(MODEL_PATH)

def evaluate():
    sequences, labels = [], []
    # สร้าง Dictionary แมปคำศัพท์กับตัวเลข (เช่น book: 0, computer: 1)
    label_map = {label:num for num, label in enumerate(actions)}
    
    print(f"📂 กำลังโหลดและแปลงข้อมูลทดสอบสำหรับ {len(actions)} คำศัพท์...")
    
    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            continue
            
        for seq_file in os.listdir(action_path):
            if seq_file.endswith(".npy"):
                res = np.load(os.path.join(action_path, seq_file))
                
                # 🌟 สกัดฟีเจอร์ใหม่: แปลงข้อมูล 378 ช่อง เป็น 392 ช่องให้เข้ากับ V5
                res_upgraded = add_angles_to_sequence(res)
                
                sequences.append(res_upgraded)
                labels.append(label_map[action])

    # 2. เตรียมข้อมูล (Padding) เติม 0 ให้ครบ 60 เฟรม
    X_test = pad_sequences(sequences, maxlen=60, padding='post', dtype='float32')
    y_true = np.array(labels)

    # 3. ให้โมเดลทำนายผล
    print(f"🧠 โมเดลกำลังวิเคราะห์ข้อมูลทั้งหมด {X_test.shape[0]} คลิป...")
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # 4. สร้าง Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 5. วาดกราฟด้วย Seaborn (ปรับแต่งสำหรับสเกลใหญ่ 50 คำ)
    plt.figure(figsize=(24, 20)) # 🌟 ขยายพื้นที่ผืนผ้าใบให้ใหญ่สะใจ
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=actions, yticklabels=actions,
                annot_kws={"size": 8}) # 🌟 ลดขนาดตัวเลขในกล่องลงจะได้ไม่ล้น
                
    plt.title('Confusion Matrix - Sign Language Interpreter (V5)', fontsize=22, pad=20)
    plt.ylabel('Actual Label (ความจริง)', fontsize=16)
    plt.xlabel('Predicted Label (โมเดลทายว่า)', fontsize=16)
    
    # 🌟 หมุนข้อความแกน X 90 องศา จะได้ไม่ขี่กัน
    plt.xticks(rotation=90, fontsize=10) 
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    # บันทึกรูปแบบความละเอียดสูง (300 DPI) เหมาะสำหรับแปะลงเล่มปริญญานิพนธ์
    save_path = os.path.join(MODEL_DIR, "confusion_matrix_v5.png")
    plt.savefig(save_path, dpi=300) 
    print(f"\n✅ วาดกราฟสำเร็จ! บันทึก Confusion Matrix ไว้ที่: {save_path}")
    
    # 6. แสดงรายงานสรุป (Precision, Recall, F1-score)
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=actions))

if __name__ == "__main__":
    evaluate()