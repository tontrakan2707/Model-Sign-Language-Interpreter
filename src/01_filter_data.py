import json
import os

def filter_top_n_classes(json_path, missing_path, top_n=50, output_file="data/filtered_top50_videos.json"):
    """
    ฟังก์ชันคัดกรองวิดีโอที่ไม่พัง และหา Top N คำศัพท์ที่มีคลิปเยอะที่สุด
    """
    print("🔍 กำลังเริ่มคัดกรองข้อมูล...")

    # 1. โหลดรายชื่อไฟล์ที่สูญหาย (Missing videos)
    missing_videos = set()
    if os.path.exists(missing_path):
        with open(missing_path, 'r') as f:
            # สมมติว่าไฟล์ missing.txt เก็บชื่อไฟล์/ID ไว้บรรทัดละ 1 ชื่อ
            missing_videos = set(line.strip() for line in f.readlines())
        print(f"⚠️ พบวิดีโอที่สูญหายใน missing.txt จำนวน: {len(missing_videos)} คลิป")
    else:
        print("⚠️ ไม่พบไฟล์ missing.txt ข้ามขั้นตอนการตัดวิดีโอเสีย")

    # 2. โหลดไฟล์ JSON หลักของ WLASL
    with open(json_path, 'r') as f:
        wlasl_data = json.load(f)

    word_stats = {}
    valid_dataset = []

    # 3. นับจำนวนคลิปที่ "ใช้งานได้จริง" ในแต่ละคำศัพท์
    for entry in wlasl_data:
        gloss = entry['gloss']
        instances = entry['instances']
        
        # กรองเฉพาะคลิปที่ไม่ได้อยู่ใน missing_videos
        valid_instances = [inst for inst in instances if inst['video_id'] not in missing_videos]
        
        if valid_instances:
            word_stats[gloss] = len(valid_instances)
            valid_dataset.append({
                "gloss": gloss,
                "instances": valid_instances
            })

    # 4. จัดเรียงเพื่อหา Top N คำศัพท์ที่มีคลิปเยอะที่สุด
    top_words = sorted(word_stats.items(), key=lambda item: item[1], reverse=True)[:top_n]
    top_glosses_set = set([word[0] for word in top_words])
    
    print(f"\n🏆 Top {top_n} คำศัพท์ที่มีคลิปสมบูรณ์มากที่สุด:")
    for i, (word, count) in enumerate(top_words, 1):
        print(f"  {i}. {word} ({count} คลิป)")

    # 5. สร้าง Dataset ใหม่ที่โฟกัสแค่ Top N
    final_filtered_data = [entry for entry in valid_dataset if entry['gloss'] in top_glosses_set]

    # บันทึกเป็นไฟล์ JSON ใหม่ เพื่อส่งต่อให้สคริปต์สกัด Keypoints
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(final_filtered_data, f, indent=4)
        
    print(f"\n✅ บันทึกข้อมูลที่คัดกรองแล้วลงใน: {output_file}")
    
    return final_filtered_data
if __name__ == "__main__":
    # กรณีที่ไฟล์ทั้งหมดถูกโหลดและแตกไฟล์ไว้ในโฟลเดอร์ data/videos/
    JSON_PATH = "../data/WLASL_v0.3.json"
    MISSING_PATH = "../data/missing.txt"
    OUTPUT_PATH = "../data/processed_data/filtered_top50_videos.json" # แยกไฟล์ผลลัพธ์มาไว้ใน processed_data เพื่อความเป็นระเบียบ
    
 
    
    filter_top_n_classes(JSON_PATH, MISSING_PATH, top_n=50, output_file=OUTPUT_PATH)