import json
import os
import shutil

# -----------------------------------------
# กำหนดเส้นทางโฟลเดอร์ให้ตรงกับโปรเจกต์ของคุณ
# -----------------------------------------
SOURCE_DIR = r'../data/videos' # โฟลเดอร์ที่เก็บวิดีโอต้นฉบับ
DEST_DIR = r'../data/renamed_videos' # โฟลเดอร์ปลายทางที่ต้องการเก็บวิดีโอที่เปลี่ยนชื่อแล้ว
JSON_PATH = '../data/WLASL_v0.3.json'
MISSING_PATH = '../data/missing.txt'

def main():
    # 1. โหลดรายชื่อวิดีโอที่หายไปจาก missing.txt เก็บไว้ใน Set เพื่อความรวดเร็วในการค้นหา
    missing_ids = set()
    if os.path.exists(MISSING_PATH):
        with open(MISSING_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                missing_ids.add(line.strip())
        print(f"Loaded {len(missing_ids)} missing video IDs.")
    else:
        print(f"Warning: {MISSING_PATH} not found.")

    # 2. โหลดข้อมูล JSON
    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found.")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        wlasl_data = json.load(f)

    # สร้างโฟลเดอร์ปลายทางหากยังไม่มี
    os.makedirs(DEST_DIR, exist_ok=True)

    copied_count = 0
    skipped_missing = 0
    not_found_count = 0

    # 3. เริ่มทำการวนลูปตามคำและคัดลอกไฟล์
    for entry in wlasl_data:
        gloss = entry['gloss'] # ชื่อคำ
        
        for instance in entry['instances']:
            video_id = str(instance['video_id'])
            
            # ตรวจสอบว่าวิดีโออยู่ใน missing.txt หรือไม่
            if video_id in missing_ids:
                skipped_missing += 1
                continue
                
            src_path = os.path.join(SOURCE_DIR, f"{video_id}.mp4")
            dest_path = os.path.join(DEST_DIR, f"{gloss}_{video_id}.mp4")
            
            # ตรวจสอบว่ามีไฟล์วิดีโอต้นฉบับอยู่จริง
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {video_id}.mp4 -> {gloss}_{video_id}.mp4")
                copied_count += 1
            else:
                not_found_count += 1

    print("-" * 30)
    print("สรุปผลการทำงาน:")
    print(f"คัดลอกสำเร็จ: {copied_count} ไฟล์")
    print(f"ข้ามไฟล์ใน missing.txt: {skipped_missing} ไฟล์")
    print(f"หาไฟล์ต้นฉบับไม่พบ: {not_found_count} ไฟล์")

if __name__ == "__main__":
    main()