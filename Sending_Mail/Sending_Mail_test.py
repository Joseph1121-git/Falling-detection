# send_email.py
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
import os, json

# 시스템 환경변수 값
# 변수명 : naver_account
# 값 : {"account":"test@naver.com", "pwd":"test", "name":"홍길동", "to":"receive@nate.com"})
ENV_NAVER = json.loads(os.environ.get('naver_account', '{}'))

# to: 받는 사람 배열
# subject: 메일 제목
# body: 메일 본문
def sendNaver(to=[], subject='[긴급]넘어짐 감지 서비스 알림!!', body='귀하의 보호자께서 넘어진 후 10초간 움직임이 없습니다. 빠르게 확인해주세요.'):
    try:

        # 네이버 접속계정 정보
        send_account    = ENV_NAVER['account']
        send_pwd        = ENV_NAVER['pwd']
        send_name       = ENV_NAVER['name']

        smtp = smtplib.SMTP_SSL('smtp.naver.com', 465)
        smtp.login(send_account, send_pwd)
        
        msg = MIMEMultipart('alternative')

        msg['Subject'] = subject
        msg['From'] = formataddr((str(Header(send_name, 'utf-8')), send_account))
        msg['To'] = ', '.join(to)

        msg.attach(MIMEText(body, 'html'))
        smtp.sendmail(send_account, to, msg.as_string())

        # 세션 종료
        smtp.quit()
        print("OK")
        return "OK"
    except Exception as ex: # 에러 종류
        print('이메일 발송 에러', ex)
        print(send_account)
        print(send_pwd)
        return ex

if __name__ == "__main__":
    sendNaver(to=[ENV_NAVER['to']])