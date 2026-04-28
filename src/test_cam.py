import cv2
cap = cv2.VideoCapture(4) # ลองเปลี่ยนเป็น 1 หรือ 2 ถ้ายังดำ
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ อ่านเฟรมไม่ได้")
        break
    cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()