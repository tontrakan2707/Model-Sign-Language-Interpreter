import os
import numpy as np
import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 1. นำเข้า ReduceLROnPlateau เพิ่มเติม
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau

from notifier import send_line_notification

DATA_PATH = "../data/processed_data/sequences"
# เปลี่ยนโฟลเดอร์เซฟเป็น v5
MODEL_DIR = "../models/saved_models/v5_expert_tuning" 
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "bilstm_model_v5.keras")
MAX_FRAMES = 60 
TOTAL_EPOCHS = 100

class ProgressNotificationCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.next_notify_percent = 10

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_epoch = epoch + 1
        current_percent = (current_epoch / self.total_epochs) * 100
        
        if current_percent >= self.next_notify_percent:
            acc = logs.get('categorical_accuracy', logs.get('accuracy', 0)) * 100
            val_acc = logs.get('val_categorical_accuracy', logs.get('val_accuracy', 0)) * 100
            
            msg = (f"⏳ [Bot] เทรนโมเดล V5 คืบหน้า: {int(self.next_notify_percent)}% "
                   f"(Epoch {current_epoch}/{self.total_epochs})\n"
                   f"📈 Train Acc: {acc:.2f}%\n"
                   f"🎯 Val Acc: {val_acc:.2f}%")
                   
            print(f"\n[ระบบกำลังส่ง LINE: คืบหน้า {int(self.next_notify_percent)}%]")
            send_line_notification(msg)
            self.next_notify_percent += 10

# 1. ฟังก์ชันคณิตศาสตร์คำนวณมุมระหว่าง 3 จุดแบบ 3 มิติ
def calculate_angle(a, b, c):
    """คำนวณมุม (องศา) ระหว่างจุด a, b, c โดยให้ b เป็นจุดยอดมุม"""
    # สร้างเวกเตอร์ ba และ bc
    ba = a - b
    bc = c - b
    
    # คำนวณ Cosine Similarity
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    
    # ป้องกันค่าที่เกินขอบเขต [-1, 1] จาก floating point error
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # แปลงเรเดียนเป็นองศา
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# 2. ฟังก์ชันแปลงพิกัดดิบ 378 จุด ให้เป็น องศาข้อต่อ
def extract_joint_angles(frame_data):
    """ดึงข้อมูล 378 ฟีเจอร์มาจัดรูปใหม่และคำนวณองศาที่สำคัญ"""
    # แยกส่วนข้อมูล (Pose 132, LH 63, RH 63, Lips 120)
    pose = frame_data[:132].reshape(33, 4)[:, :3] # เอาเฉพาะ x,y,z ตัด v ทิ้ง
    lh = frame_data[132:195].reshape(21, 3)
    rh = frame_data[195:258].reshape(21, 3)
    
    angles = []
    
    # --- องศาของแขน (Pose) ---
    # 11: ไหล่ซ้าย, 13: ศอกซ้าย, 15: มือซ้าย | 12: ไหล่ขวา, 14: ศอกขวา, 16: มือขวา
    # 23: สะโพกซ้าย, 24: สะโพกขวา
    if np.any(pose): # ตรวจสอบว่ามีข้อมูลคนหรือไม่
        angles.append(calculate_angle(pose[11], pose[13], pose[15])) # ศอกซ้ายพับแค่ไหน
        angles.append(calculate_angle(pose[12], pose[14], pose[16])) # ศอกขวาพับแค่ไหน
        angles.append(calculate_angle(pose[23], pose[11], pose[13])) # ยกแขนซ้ายสูงแค่ไหน
        angles.append(calculate_angle(pose[24], pose[12], pose[14])) # ยกแขนขวาสูงแค่ไหน
    else:
        angles.extend([0]*4)
        
    # --- องศาของมือซ้าย (Left Hand) ---
    # 0: ข้อมือ, 5: โคนนิ้วชี้, 8: ปลายนิ้วชี้ | ทำเพื่อเช็คว่ากำมือหรือแบมือ
    if np.any(lh):
        angles.append(calculate_angle(lh[0], lh[1], lh[4])) # นิ้วโป้ง
        angles.append(calculate_angle(lh[0], lh[5], lh[8])) # นิ้วชี้
        angles.append(calculate_angle(lh[0], lh[9], lh[12])) # นิ้วกลาง
        angles.append(calculate_angle(lh[0], lh[13], lh[16])) # นิ้วนาง
        angles.append(calculate_angle(lh[0], lh[17], lh[20])) # นิ้วก้อย
    else:
        angles.extend([0]*5)
        
    # --- องศาของมือขวา (Right Hand) ---
    if np.any(rh):
        angles.append(calculate_angle(rh[0], rh[1], rh[4])) # นิ้วโป้ง
        angles.append(calculate_angle(rh[0], rh[5], rh[8])) # นิ้วชี้
        angles.append(calculate_angle(rh[0], rh[9], rh[12])) # นิ้วกลาง
        angles.append(calculate_angle(rh[0], rh[13], rh[16])) # นิ้วนาง
        angles.append(calculate_angle(rh[0], rh[17], rh[20])) # นิ้วก้อย
    else:
        angles.extend([0]*5)
        
    # รวมองศา 14 ค่าที่คำนวณได้ Normalize ให้มีค่าอยู่ในช่วง 0-1 (หาร 180)
    normalized_angles = np.array(angles) / 180.0
    
    # นำไปต่อท้ายฟีเจอร์เดิม 378 จุด -> กลายเป็น 392 ฟีเจอร์
    return np.concatenate([frame_data, normalized_angles])

# 3. ฟังก์ชันปรับปรุงข้อมูลทั้งหมด (เรียกใช้ตอนโหลดข้อมูล)
def apply_feature_engineering(sequences):
    print("🧠 กำลังทำ Feature Engineering (คำนวณสมการองศาข้อต่อ)...")
    engineered_sequences = []
    for seq in sequences:
        new_seq = []
        for frame in seq:
            new_seq.append(extract_joint_angles(frame))
        engineered_sequences.append(np.array(new_seq))
        
    return engineered_sequences

def load_data(data_path):
    print("📂 กำลังโหลดข้อมูล Sequences...")
    sequences, labels = [], []
    actions = np.array(os.listdir(data_path))
    label_map = {label:num for num, label in enumerate(actions)}
    
    for action in actions:
        action_path = os.path.join(data_path, action)
        for sequence_file in os.listdir(action_path):
            if sequence_file.endswith(".npy"):
                res = np.load(os.path.join(action_path, sequence_file))
                sequences.append(res)
                labels.append(label_map[action])
                
    return sequences, labels, actions

# 2. อัปเกรดฟังก์ชัน Augmentation (เพิ่มการซูมภาพ)
def augment_coordinates_advanced(X_train, y_train):
    print("🧬 กำลังทำ Advanced Augmentation (Noise + Scaling)...")
    X_aug, y_aug = [], []
    for x, y in zip(X_train, y_train):
        X_aug.append(x) # 1. เก็บต้นฉบับ
        y_aug.append(y)
        
        # 2. สร้างแฝด: เติม Noise สั่นๆ + สุ่มซูมเข้า/ออก (85% - 115%)
        noise = np.random.normal(0, 0.01, x.shape)
        scale_factor = np.random.uniform(0.85, 1.15)
        x_advanced = (x * scale_factor) + noise
        
        X_aug.append(x_advanced)
        y_aug.append(y)
        
    return np.array(X_aug), np.array(y_aug)

def build_and_train_model():
    sequences, labels, actions = load_data(DATA_PATH)
    
    sequences_engineered = apply_feature_engineering(sequences)
    
    X = pad_sequences(sequences_engineered, maxlen=MAX_FRAMES, padding='post', dtype='float32')
    y = to_categorical(labels).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_aug, y_train_aug = augment_coordinates_advanced(X_train, y_train)

    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh'), input_shape=(MAX_FRAMES, 392)))
    model.add(Dropout(0.4)) 
    model.add(LSTM(128, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(actions.shape[0], activation='softmax')) 
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # 3. เพิ่มไม้ตาย: ReduceLROnPlateau
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) # ยืดความอดทนเป็น 20
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    line_notifier = ProgressNotificationCallback(total_epochs=TOTAL_EPOCHS)

    print("\n🚀 กำลังเริ่มเทรนโมเดล V5 (Expert Tuning)...")
    history = model.fit(
        X_train_aug, y_train_aug, 
        epochs=TOTAL_EPOCHS, 
        batch_size=16, 
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr, line_notifier] # ใส่ reduce_lr ลงใน callbacks
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    np.save(os.path.join(MODEL_DIR, "actions.npy"), actions)
    
    acc_key = 'categorical_accuracy' if 'categorical_accuracy' in history.history else 'accuracy'
    val_acc_key = 'val_categorical_accuracy' if 'val_categorical_accuracy' in history.history else 'val_accuracy'
    
    final_acc = history.history[acc_key][-1] * 100
    val_acc = history.history[val_acc_key][-1] * 100
    
    msg = (f"🎉 [Bot] เทรนโมเดล V5 ทะลุขีดจำกัดสำเร็จ!\n"
           f"📈 Train Acc: {final_acc:.2f}%\n"
           f"🎯 Val Acc: {val_acc:.2f}%\n"
           f"💾 บันทึกที่ v5_expert_tuning")
    print("\n[ระบบกำลังส่ง LINE: รายงานผลตอนจบ]")
    send_line_notification(msg)

if __name__ == "__main__":
    try:
        print("\n[ระบบกำลังส่ง LINE: แจ้งเตือนเริ่มทำงาน]")
        send_line_notification("⏳ [Bot] ลุยเทรนโมเดล V5 (เป้าหมาย 80%+) ครับ!")
        build_and_train_model()
    except Exception as e:
        error_msg = f"❌ [Bot] เทรนล้มเหลว: {e}"
        print(error_msg)
        send_line_notification(error_msg)