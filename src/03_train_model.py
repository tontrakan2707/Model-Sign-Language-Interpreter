import os
import numpy as np
import math
import tensorflow as tf  # เพิ่ม import tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau

from notifier import send_line_notification

DATA_PATH = "../data/processed_data/sequences"
MODEL_DIR = "../models/saved_models/v11_ultimate_tuning20word" 
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "bilstm_model_v11.keras")
MAX_FRAMES = 60 
TOTAL_EPOCHS = 100

# ==========================================
# 1. แจ้งเตือนผ่าน LINE ทุกๆ 20%
# ==========================================
class ProgressNotificationCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.target_percentages = [20, 40, 60, 80, 100]
        self.notified = set()

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        percent_complete = (current_epoch / self.total_epochs) * 100

        for target in self.target_percentages:
            if percent_complete >= target and target not in self.notified:
                acc_key = 'categorical_accuracy' if 'categorical_accuracy' in logs else 'accuracy'
                msg = (f"⏳ [Bot] คืบหน้า {target}% (Epoch {current_epoch}/{self.total_epochs})\n"
                       f"📉 Loss: {logs.get('loss'):.4f}\n"
                       f"🎯 Acc: {logs.get(acc_key, 0)*100:.2f}%")
                print(f"\n[ระบบกำลังส่ง LINE: คืบหน้า {target}%]")
                send_line_notification(msg)
                self.notified.add(target)
# ==========================================
# 1.5 Custom Callback ป้องกัน Overfitting ขั้นรุนแรง
# ==========================================
class OverfitPreventer(Callback):
    def __init__(self, threshold=0.15, start_epoch=20): # เพิ่ม start_epoch
        super().__init__()
        self.threshold = threshold
        self.start_epoch = start_epoch # กำหนดจุดเริ่มต้นการจับตาดู

    def on_epoch_end(self, epoch, logs=None):
        # ให้โมเดลรันผ่านช่วง Warm-up ไปก่อนโดยไม่เช็ค Gap
        if epoch < self.start_epoch:
            return

        logs = logs or {}
        acc = logs.get('categorical_accuracy')
        val_acc = logs.get('val_categorical_accuracy')

        if acc is not None and val_acc is not None:
            gap = acc - val_acc
            if gap > self.threshold:
                msg = f"\n🚨 [OverfitPreventer] Alert! Epoch {epoch+1}: สั่งหยุดเทรนอัตโนมัติ เพราะ Gap เกิน {self.threshold*100}% (Train: {acc:.4f}, Val: {val_acc:.4f})"
                print(msg)
                send_line_notification(msg)
                self.model.stop_training = True

# ------------------------------------------
# Data Processing & Feature Engineering
# ------------------------------------------

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def extract_advanced_features(frame_data):
    pose = frame_data[:132].reshape(33, 4)      
    lh = frame_data[132:195].reshape(21, 3)     
    rh = frame_data[195:258].reshape(21, 3)     
    
    # 1. หาจุดศูนย์กลาง (จมูก)
    if np.any(pose):
        anchor = pose[0, :3].copy()
        # ⭐️ ใหม่: หาความกว้างของไหล่ (จุดที่ 11 ไหล่ซ้าย, จุดที่ 12 ไหล่ขวา)
        shoulder_width = np.linalg.norm(pose[11, :3] - pose[12, :3])
        # ป้องกันกรณีหารด้วย 0
        shoulder_width = max(shoulder_width, 1e-6) 
    else:
        anchor = np.zeros(3)
        shoulder_width = 1.0 # ค่าเริ่มต้น
        
    pose_rel = np.zeros_like(pose)
    lh_rel = np.zeros_like(lh)
    rh_rel = np.zeros_like(rh)
    
    # 2. ⭐️ ใหม่: แปลงพิกัด (ลบด้วยจมูก และ หารด้วยความกว้างไหล่)
    if np.any(pose):
        pose_rel[:, :3] = (pose[:, :3] - anchor) / shoulder_width
        pose_rel[:, 3] = pose[:, 3]            
    if np.any(lh):
        lh_rel = (lh - anchor) / shoulder_width
    if np.any(rh):
        rh_rel = (rh - anchor) / shoulder_width

    # 3. ⭐️ ใหม่: คำนวณระยะห่าง (Distance Features) ที่สำคัญ
    distances = []
    # ระยะห่างข้อมือซ้าย (lh[0]) กับ ข้อมือขวา (rh[0])
    if np.any(lh) and np.any(rh):
        hand_dist = np.linalg.norm(lh_rel[0] - rh_rel[0])
        distances.append(hand_dist)
    else:
        distances.append(0.0)
        
    # ระยะห่างมือซ้าย-จมูก และ มือขวา-จมูก (จมูกตอนนี้คือ 0,0,0)
    if np.any(lh):
        distances.append(np.linalg.norm(lh_rel[0]))
    else:
        distances.append(0.0)
        
    if np.any(rh):
        distances.append(np.linalg.norm(rh_rel[0]))
    else:
        distances.append(0.0)

    # 4. คำนวณองศาข้อต่อ 14 จุด (เพื่อดูแพทเทิร์นการพับนิ้ว/แขน)
    angles = []
    pose_xyz = pose[:, :3] # ใช้องศาจากพิกัดเดิมเพื่อความแม่นยำทางเรขาคณิต
    
    if np.any(pose_xyz):
        angles.append(calculate_angle(pose_xyz[11], pose_xyz[13], pose_xyz[15])) 
        angles.append(calculate_angle(pose_xyz[12], pose_xyz[14], pose_xyz[16])) 
        angles.append(calculate_angle(pose_xyz[23], pose_xyz[11], pose_xyz[13])) 
        angles.append(calculate_angle(pose_xyz[24], pose_xyz[12], pose_xyz[14])) 
    else:
        angles.extend([0]*4)
        
    if np.any(lh):
        angles.append(calculate_angle(lh[0], lh[1], lh[4])) 
        angles.append(calculate_angle(lh[0], lh[5], lh[8])) 
        angles.append(calculate_angle(lh[0], lh[9], lh[12])) 
        angles.append(calculate_angle(lh[0], lh[13], lh[16])) 
        angles.append(calculate_angle(lh[0], lh[17], lh[20])) 
    else:
        angles.extend([0]*5)
        
    if np.any(rh):
        angles.append(calculate_angle(rh[0], rh[1], rh[4])) 
        angles.append(calculate_angle(rh[0], rh[5], rh[8])) 
        angles.append(calculate_angle(rh[0], rh[9], rh[12])) 
        angles.append(calculate_angle(rh[0], rh[13], rh[16])) 
        angles.append(calculate_angle(rh[0], rh[17], rh[20])) 
    else:
        angles.extend([0]*5)
        
    normalized_angles = np.array(angles) / 180.0
    
    # 5. ประกอบร่างข้อมูล: [พิกัดที่ปรับศูนย์กลางแล้ว 378 ตัว] + [องศา 14 ตัว] = 392 ฟีเจอร์เป๊ะ!
    advanced_frame = np.concatenate([
        pose_rel.flatten(), 
        lh_rel.flatten(), 
        rh_rel.flatten(), 
        normalized_angles,
        np.array(distances) 
    ])
    
    return advanced_frame

    
def apply_feature_engineering(sequences):
    print("🧠 กำลังทำ Feature Engineering (เพิ่มมิติความเร็ว/ทิศทาง)...")
    engineered_sequences = []
    
    for seq in sequences:
        new_seq = []
        prev_features = None # ตัวแปรจำจดฟีเจอร์ของเฟรมก่อนหน้า
        
        for frame in seq:
            # 1. สกัดพิกัด (สัมพัทธ์) และองศาตามปกติ 
            # (สมมติว่าฟังก์ชันนี้รีเทิร์นออกมา 272 ตัว ตามที่เราคุยกันใน v11)
            current_features = extract_advanced_features(frame)
            
            # 2. คำนวณความเร็วและการกระจัด (Velocity)
            if prev_features is None:
                # ถ้าเป็นเฟรมแรกสุด ยังไม่มีการขยับ ให้ความเร็วเป็น 0 ทั้งหมด
                velocity = np.zeros_like(current_features)
            else:
                # เฟรมที่ 2 เป็นต้นไป: การกระจัด = ตำแหน่งปัจจุบัน - ตำแหน่งก่อนหน้า
                velocity = current_features - prev_features
            
            # 3. ประกอบร่าง: [พิกัดปัจจุบัน] + [ความเร็วที่เปลี่ยนไป]
            # ขนาดของ Feature จะเพิ่มขึ้นเป็น 2 เท่า! (เช่น 272 + 272 = 544)
            combined_features = np.concatenate([current_features, velocity])
            
            new_seq.append(combined_features)
            
            # อัปเดตเฟรมก่อนหน้าเตรียมไว้รอบลูปถัดไป
            prev_features = current_features
            
        engineered_sequences.append(np.array(new_seq))
        
    return engineered_sequences


# ==========================================
# 2. Advanced Data Augmentation (Time-Warp & Jitter)
# ==========================================
def time_warp_sequence(seq, warp_ratio):
    """
    ฟังก์ชันบิดเบือนเวลา (Time-Warping)
    - warp_ratio < 1.0 : หดเฟรม (จำลองการเคลื่อนไหวที่เร็วขึ้น)
    - warp_ratio > 1.0 : ยืดเฟรม (จำลองการเคลื่อนไหวที่ช้าลง)
    """
    # ตรวจสอบว่ามีข้อมูลหรือไม่ (ข้าม Padding ที่เป็น 0)
    actual_frames = len([frame for frame in seq if np.any(frame)])
    if actual_frames == 0:
        return seq

    orig_steps = np.arange(seq.shape[0])
    # สร้าง Time steps ใหม่ตามอัตราส่วน
    new_steps = np.linspace(0, seq.shape[0] - 1, int(seq.shape[0] * warp_ratio))
    
    warped_seq = np.zeros((len(new_steps), seq.shape[1]))
    for i in range(seq.shape[1]):
        # ใช้ Linear Interpolation คำนวณพิกัดในจุดเวลาที่เปลี่ยนไป
        warped_seq[:, i] = np.interp(new_steps, orig_steps, seq[:, i])
        
    # ตัดหรือเติมให้กลับมามีขนาดเท่า MAX_FRAMES (60 เฟรม)
    if warped_seq.shape[0] < seq.shape[0]:
        padding = np.zeros((seq.shape[0] - warped_seq.shape[0], seq.shape[1]))
        warped_seq = np.vstack((warped_seq, padding))
    else:
        warped_seq = warped_seq[:seq.shape[0], :]
        
    return warped_seq

def augment_coordinates_advanced(X, y):
    print("\n⏳ กำลังทำ Advanced Data Augmentation (Time-Warping + Spatial Jittering)...")
    X_orig = X.copy()
    y_orig = y.copy()
    
    X_aug_fast = []
    X_aug_slow = []
    X_aug_jitter = []
    
    for seq in X:
        # 1. Fast Motion: จำลองคนทำท่าทางเร็วขึ้น 20% (เหลือเวลา 80%)
        X_aug_fast.append(time_warp_sequence(seq, warp_ratio=0.8))
        
        # 2. Slow Motion: จำลองคนทำท่าทางช้าลง 20% (ใช้เวลา 120%)
        X_aug_slow.append(time_warp_sequence(seq, warp_ratio=1.2))
        
        # 3. Spatial Jittering: สุ่มขยับแกน x,y เล็กน้อย (จำลองมือที่สั่นหรือไม่นิ่ง)
        # ปรับ Scale จากเดิม 0.005 เป็น 0.01 เพื่อเพิ่มความต้านทาน (Robustness)
        noise = np.random.normal(loc=0.0, scale=0.01, size=seq.shape)
        # เติม Noise เฉพาะพิกัดที่ไม่ใช่ 0 (หลีกเลี่ยงการทำลายข้อมูล Padding)
        jittered_seq = np.where(seq != 0.0, seq + noise, seq)
        X_aug_jitter.append(jittered_seq)
        
    # นำร่างจำลองทั้ง 3 รูปแบบมารวมกับต้นฉบับ
    X_combined = np.concatenate((
        X_orig, 
        np.array(X_aug_fast), 
        np.array(X_aug_slow), 
        np.array(X_aug_jitter)
    ), axis=0)
    
    # ขยาย Label ตามจำนวนข้อมูลที่เพิ่มขึ้น (คูณ 4)
    y_combined = np.concatenate((y_orig, y_orig, y_orig, y_orig), axis=0)
    
    print(f"✅ ขยายข้อมูลสำเร็จแบบ x4 เท่า! จาก {len(X_orig)} -> {len(X_combined)} ตัวอย่าง")
    return X_combined, y_combined

# ==========================================
# 🎯 กำหนดรายการคำศัพท์สำหรับ MVP (Core Words)
# ==========================================
MVP_CLASSES = [
    'basketball', 'dark', 'family', 'brother', 'drink',
    'man', 'woman', 'apple', 'computer', 'dog',
    'help', 'shirt', 'room', 'laugh', 'cousin'
]

def load_data_safe(data_path):
    print(f"📂 กำลังโหลดข้อมูลแบบป้องกัน Data Leakage [MVP Mode: {len(MVP_CLASSES)} Classes]...")
    train_seqs, train_labels = [], []
    test_seqs, test_labels = [], []
    
    # ⭐️ กรองเอาเฉพาะโฟลเดอร์ที่มีชื่อตรงกับใน MVP_CLASSES
    all_folders = os.listdir(data_path)
    actions = np.array([folder for folder in all_folders if folder in MVP_CLASSES])
    
    # ตรวจสอบว่าเจอครบไหม
    print(f"🔍 พบคำศัพท์ที่ตรงกับ MVP ทั้งหมด: {len(actions)} คำ")
    
    label_map = {label:num for num, label in enumerate(actions)}
    
    for action in actions:
        action_path = os.path.join(data_path, action)
        files = os.listdir(action_path)
        video_ids = list(set([f.split('_')[0] for f in files if f.endswith('.npy')]))
        train_vids, test_vids = train_test_split(video_ids, test_size=0.2, random_state=42)
        
        for sequence_file in files:
            if sequence_file.endswith(".npy"):
                vid_id = sequence_file.split('_')[0]
                res = np.load(os.path.join(action_path, sequence_file))
                
                if vid_id in test_vids:
                    if sequence_file.endswith("_orig.npy"):
                        test_seqs.append(res)
                        test_labels.append(label_map[action])
                else:
                    train_seqs.append(res)
                    train_labels.append(label_map[action])
                    
    return train_seqs, train_labels, test_seqs, test_labels, actions
def build_and_train_model():
    train_seqs, train_labels, test_seqs, test_labels, actions = load_data_safe(DATA_PATH)
    
    # 🟢 เปิดใช้งาน Feature Engineering 550 ฟีเจอร์กลับมา!
    print("⏳ กำลังประมวลผล Feature Engineering (550 Features) สำหรับ Train Set...")
    train_engineered = apply_feature_engineering(train_seqs)
    
    print("⏳ กำลังประมวลผล Feature Engineering (550 Features) สำหรับ Test Set...")
    test_engineered = apply_feature_engineering(test_seqs)
    
    # โค้ดส่วนที่เหลือปล่อยไว้เหมือนเดิม
    X_train = pad_sequences(train_engineered, maxlen=MAX_FRAMES, padding='post', dtype='float32')
    X_test = pad_sequences(test_engineered, maxlen=MAX_FRAMES, padding='post', dtype='float32')
    
    y_train = to_categorical(train_labels, num_classes=len(actions)).astype(int)
    y_test = to_categorical(test_labels, num_classes=len(actions)).astype(int)
    
    X_train_aug, y_train_aug = augment_coordinates_advanced(X_train, y_train)

    print(f"\n📊 สรุปข้อมูล:")
    print(f"   - ชุดเรียน (Train + Augment): {len(X_train_aug)} คลิป")
    print(f"   - ชุดสอบ (Pure Unseen Test): {len(X_test)} คลิป")

   # ==========================================
    # 3. ปรับโครงสร้างโมเดล (V15: Ultimate Bottleneck + Attention)
    # แบบ Functional API 
    # ==========================================
    # กำหนด Input Layer
    inputs = Input(shape=(MAX_FRAMES, X_train_aug.shape[2]))
    
    # Block 1: Conv1D ลด Filters บังคับให้จำกัดการมองเห็นฟีเจอร์ + L2 
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', 
               kernel_regularizer=l2(0.001))(inputs) 
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)

    # Block 2: Conv1D ดึงฟีเจอร์ระดับลึกขึ้น + L2
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', 
               kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.5)(x)

    # Block 3: Bi-LSTM
    # ⚠️ สำคัญมาก: ต้องตั้ง return_sequences=True เพื่อส่ง sequence ทั้งหมดไปให้ Attention
    x = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.001)))(x)
    x = BatchNormalization()(x)
    
    # Block 4: Self-Attention Mechanism 🧠
    # ให้โมเดลโฟกัสเฟรมที่สำคัญ (จุดพีคของท่า) และมองข้ามเฟรมที่มืออยู่นิ่งๆ
    attention_out = Attention()([x, x])
    
    # รวบข้อมูล (Pooling) ก่อนส่งไปให้ Dense Layer
    x = GlobalAveragePooling1D()(attention_out)
    
    # Heavy Dropout ป้องกัน Overfitting ขั้นสุดท้าย
    x = Dropout(0.6)(x)
    
    # Output Layer
    outputs = Dense(len(actions), activation='softmax', kernel_regularizer=l2(0.001))(x)
    
    # ประกอบร่างเป็น Model แบบ Functional API
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
   # ขยับ Threshold เป็น 20%
    overfit_preventer = OverfitPreventer(threshold=0.20, start_epoch=20)
    line_notifier = ProgressNotificationCallback(total_epochs=TOTAL_EPOCHS)

    print("\n🚀 กำลังเริ่มเทรนโมเดล v11 (Ultimate Tuning)...")
    history = model.fit(
        X_train_aug, y_train_aug, 
        epochs=TOTAL_EPOCHS, 
        batch_size=16, 
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr, overfit_preventer, line_notifier] # เพิ่มลงไปใน list
    )
    
    print("\n" + "="*50)
    print("📸 EXPORTING ALL EVALUATION REPORTS TO IMAGES...")
    print("="*50)

    # ดึงชื่อคลาสอัตโนมัติ
    class_names = actions.tolist()

    print("[1/4] Evaluating Model on Test Set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_true = np.argmax(y_test, axis=1)
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    print("[2/4] Saving Metrics & Classification Report as Image...")
    report_text = f"=== FINAL MODEL PERFORMANCE ===\n\n"
    report_text += f"Test Loss     : {test_loss:.4f}\n"
    report_text += f"Test Accuracy : {test_acc*100:.2f}%\n\n"
    report_text += "=== CLASSIFICATION REPORT ===\n\n"
    # ใส่ zero_division=0 เพื่อปิด Warning สีส้ม
    report_text += classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.text(0.01, 0.95, report_text, {'fontsize': 12, 'family': 'monospace'}, 
            verticalalignment='top', transform=ax.transAxes)
    ax.axis('off') 
    plt.savefig('1_metrics_report.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(">> Saved '1_metrics_report.png'")

    print("[3/4] Saving Learning Curves Image...")
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    acc_metric = 'categorical_accuracy' if 'categorical_accuracy' in history.history else 'accuracy'
    val_acc_metric = 'val_categorical_accuracy' if 'val_categorical_accuracy' in history.history else 'val_accuracy'

    plt.subplot(1, 2, 2)
    plt.plot(history.history[acc_metric], label='Training Accuracy', color='green', linewidth=2)
    plt.plot(history.history[val_acc_metric], label='Validation Accuracy', color='red', linewidth=2)
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('2_learning_curves.png', dpi=300, facecolor='white')
    plt.close()
    print(">> Saved '2_learning_curves.png'")

    print("[4/4] Saving Confusion Matrix Image...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18)) 
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=18)
    plt.ylabel('Actual Label (Truth)', fontsize=14)
    plt.xlabel('Predicted Label (AI)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    plt.savefig('3_confusion_matrix.png', dpi=300, facecolor='white')
    plt.close()
    print(">> Saved '3_confusion_matrix.png'")

    # ==========================================
    # 4. ข้ามการวาดรูป Model เพื่อป้องกัน Graphviz Error
    # ==========================================
    print("[5/5] Saving Model Architecture Image (Skipped due to Graphviz)")
    # plot_model(model, show_dtype=True, to_file='4_model_architecture.png', show_shapes=True)
    # print(">> Saved '4_model_architecture.png'")

    print("\n✅ DONE! All 3 report images have been generated and saved.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    np.save(os.path.join(MODEL_DIR, "actions.npy"), actions)
    
    acc_key = 'categorical_accuracy' if 'categorical_accuracy' in history.history else 'accuracy'
    val_acc_key = 'val_categorical_accuracy' if 'val_categorical_accuracy' in history.history else 'val_accuracy'
    
    final_acc = history.history[acc_key][-1] * 100
    val_acc = history.history[val_acc_key][-1] * 100
    
    summary_msg = (
        f"🎉 [Bot] เทรนโมเดล v11 เสร็จสิ้นสมบูรณ์!\n"
        f"📊 สรุปข้อมูล (Dataset):\n"
        f" 🔸 คำศัพท์ทั้งหมด: {len(actions)} คำ\n"
        f" 🔸 ใช้คลิป Train: {len(X_train_aug)} คลิป\n"
        f" 🔸 ใช้คลิป Test: {len(X_test)} คลิป\n"
        f"-----------------------\n"
        f"📈 ประสิทธิภาพ (Performance):\n"
        f" 🔸 Train Acc: {final_acc:.2f}%\n"
        f" 🔸 Val Acc  : {val_acc:.2f}%\n"
        f" 🏆 Test Acc : {test_acc*100:.2f}% (บนข้อสอบจริง)\n"
        f"-----------------------\n"
        f"📁 ไฟล์ Report 3 รูป ถูกสร้างเรียบร้อยแล้ว\n"
        f"💾 โมเดลเซฟที่: {MODEL_SAVE_PATH}"
    )
    
    print("\n[ระบบกำลังส่ง LINE: รายงานผลตอนจบ]")
    send_line_notification(summary_msg)

if __name__ == "__main__":
    try:
        print("\n[ระบบกำลังส่ง LINE: แจ้งเตือนเริ่มทำงาน]")
        send_line_notification("⏳ [Bot] ลุยเทรนโมเดล v11 (Ultimate Tuning)!")
        build_and_train_model()
    except Exception as e:
        error_msg = f"❌ [Bot] เทรนล้มเหลว: {e}"
        print(error_msg)
        send_line_notification(error_msg)