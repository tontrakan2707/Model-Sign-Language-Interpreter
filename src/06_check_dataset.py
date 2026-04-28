import json
import os
import numpy as np

# 1. เช็คจากไฟล์ JSON (เพื่อให้รู้ชื่อวิดีโอต้นฉบับ)
JSON_PATH = "../data/processed_data/filtered_top50_videos.json"

# 2. เช็คจากโมเดล (เพื่อให้ชัวร์ว่าโมเดลจำคำพวกนี้ได้จริงๆ)
ACTIONS_PATH = "../models/saved_models/v5_expert_tuning/actions.npy"

print("==================================================")
print(" 📋 รายงานข้อมูลคำศัพท์และไฟล์วิดีโอ")
print("==================================================")

# โหลดดูคำศัพท์ที่โมเดลเทรนผ่าน
if os.path.exists(ACTIONS_PATH):
    actions = np.load(ACTIONS_PATH)
    print(f"🧠 โมเดล V5 ของเราเรียนรู้คำศัพท์ไปทั้งหมด: {len(actions)} คำ\n")
else:
    print("❌ ไม่พบไฟล์ actions.npy ของโมเดล\n")

# โหลดดูรายละเอียดไฟล์วิดีโอ
if os.path.exists(JSON_PATH):
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"📂 รายละเอียดไฟล์วิดีโอต้นฉบับ (จาก JSON):")
    for idx, entry in enumerate(data):
        gloss = entry['gloss']
        instances = entry['instances']
        
        # ดึงชื่อไฟล์ video_id ทั้งหมดของคำนี้มาใส่ List
        video_files = [f"{inst['video_id']}.mp4" for inst in instances]
        
        print(f"{idx + 1:02d}. คำศัพท์: [{gloss}]")
        print(f"    🎬 จำนวน: {len(video_files)} คลิป")
        print(f"    📁 รายชื่อไฟล์: {', '.join(video_files[:5])}" + ("..." if len(video_files) > 5 else ""))
        print("-" * 50)
else:
    print(f"❌ ไม่พบไฟล์ {JSON_PATH}")

print("\n💡 Tip: คุณสามารถก๊อปปี้ 'รายชื่อไฟล์' ด้านบน ไปใส่ในตัวแปร TEST_VIDEO_PATH ในไฟล์ 05_video_detect.py เพื่อทดสอบได้เลยครับ!")