import requests
import json

def send_line_notification(message):
    """
    ฟังก์ชันสำหรับส่งแจ้งเตือนเข้า LINE ผ่าน Messaging API
    """
    # ⚠️ ใส่ค่าของคุณที่ได้จาก LINE Developers Console ลงไป
    CHANNEL_ACCESS_TOKEN = '-- ใส่ Channel Access Token ของคุณที่ได้จาก LINE Developers Console --'
    USER_ID = '-- ใส่ User ID ของคุณที่ต้องการส่งข้อความไปหา --'
    
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {CHANNEL_ACCESS_TOKEN}'
    }
    payload = {
        'to': USER_ID,
        'messages': [
            {
                'type': 'text',
                'text': message
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print("📲 ส่งการแจ้งเตือนเข้า LINE สำเร็จ!")
        else:
            print(f"❌ ส่งการแจ้งเตือน LINE ไม่สำเร็จ. HTTP Status: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการส่ง LINE: {e}")

# --- ตัวอย่างการทดสอบ ---
if __name__ == "__main__":
    send_line_notification("🤖 ทดสอบระบบแจ้งเตือนจากบอท ELT")