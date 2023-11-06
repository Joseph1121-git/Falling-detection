import smtplib
import os
import json
import io
import tkinter as tk
from PIL import ImageGrab, ImageTk
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.header import Header
from email.utils import formataddr

# Constants
SMTP_SERVER = 'smtp.naver.com'
SMTP_PORT = 465
ENV_KEY = 'naver_account'
IMAGE_PATH = r"C:\Users\황명주\OneDrive\바탕 화면\Bakjun\falling\Falling-detection\img2-1.jpg"

# Load environment variables
ENV_NAVER = json.loads(os.environ.get('naver_account', '{}'))

def sendNaver(to=[], subject='[긴급]넘어짐 감지 서비스 알림!!', body='귀하의 보호자께서 넘어진 후 10초간 움직임이 없습니다. 빠르게 확인해주세요.', capture_screen=False):
    try:
        send_account = ENV_NAVER['account']
        send_pwd = ENV_NAVER['pwd']
        send_name = ENV_NAVER['name']

        smtp = smtplib.SMTP_SSL('smtp.naver.com', 465)
        smtp.login(send_account, send_pwd)

        msg = MIMEMultipart('alternative')

        msg['Subject'] = subject
        msg['From'] = formataddr((str(Header(send_name, 'utf-8')), send_account))
        msg['To'] = ', '.join(to)

        msg.attach(MIMEText(body, 'html'))

        # 화면 캡쳐가 필요한 경우
        if capture_screen:
            img = ImageGrab.grab()
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            img_data = buf.getvalue()

            image = MIMEImage(img_data, name="screenshot.png")
            msg.attach(image)

        smtp.sendmail(send_account, to, msg.as_string())

        # 세션 종료
        smtp.quit()
        print("OK")
        return "OK"
    except Exception as ex:
        print('이메일 발송 에러', ex)
        print(send_account)
        print(send_pwd)
        return ex

def main():
    root = tk.Tk()
    root.title('Enter Recipient Email')
    root.attributes('-fullscreen', True)

    img = ImageTk.PhotoImage(file=IMAGE_PATH)
    panel = tk.Label(root, image=img)
    panel.place(relwidth=1, relheight=1)

    frame = tk.Frame(root, bg='white')
    frame.place(relx=0.5, rely=0.5, anchor='center', y=105)

    tk.Label(frame, text="보호자분의 이메일을 입력해주세요:", bg='white', font=('Arial', 16)).pack()
    email_entry = tk.Entry(frame, width=30, font=('Arial', 14))
    email_entry.pack()

    def submit():
        email_address = email_entry.get()
        if email_address:
            sendNaver(to=[email_address], capture_screen=True)
        root.destroy()

    tk.Button(frame, text='확인', command=submit, bg='#0066FF', fg='white', width=10, height=1, font=('Arial', 14)).pack()
    root.mainloop()

if __name__ == "__main__":
    main()
    
