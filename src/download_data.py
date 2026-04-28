import os
import subprocess

def download_dataset():
    # ตั้งค่า Environment Variable ผ่านโค้ด Python
    os.environ['KAGGLE_API_TOKEN'] = "-- ใส่ API Token ของคุณที่ได้จาก Kaggle --" # WLASL (World Level American Sign Language) Video Dataset ที่เราจะใช้เป็นฐานข้อมูลในการเทรนโมเดลของเรา https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
    
    dataset_id = "risangbaskoro/wlasl-processed"
    target_dir = "data"

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    print("🚀 กำลังเริ่มดาวน์โหลด Dataset...")
    try:
        # ใช้คำสั่ง kaggle ผ่าน subprocess
        # หมายเหตุ: ต้องติดตั้ง pip install kaggle ก่อนนะ
        subprocess.run([
            'kaggle', 'datasets', 'download', 
            '-d', dataset_id, 
            '-p', target_dir, 
            '--unzip'
        ], check=True)
        print("✅ ดาวน์โหลดและแตกไฟล์เรียบร้อยในโฟลเดอร์ /data")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    download_dataset()