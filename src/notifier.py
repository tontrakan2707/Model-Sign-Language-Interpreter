import requests
import json

def send_line_notification(message):
    """
    ฟังก์ชันสำหรับส่งแจ้งเตือนเข้า LINE ผ่าน Messaging API
    """
    # ⚠️ ใส่ค่าของคุณที่ได้จาก LINE Developers Console ลงไป
    CHANNEL_ACCESS_TOKEN = 'ukv6aFA4BMvSLBiapQqOg87nFVlFNIWfU51wXhcFXGBYW5RX6mnZxz9omf85sWeOLiK3kqnbJbNG8wv7OP6TivkG7Z92GmrPAn8EOqtavjySp24RMGoSY48lDq5aaP298JeHgFrPdJFnRDCleYQ2egdB04t89/1O/w1cDnyilFU='
    USER_ID = 'Uff3badc146018557cf25e1718e734a33'
    
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