# import vlc
# import time

# # VLC 인스턴스 생성
# player = vlc.MediaPlayer(r"C:\Users\황명주\OneDrive\바탕 화면\Bakjun\falling\Falling-detection\Alarm.mp4")

# player.play()

# # 재생이 시작되었는지 확인
# time.sleep(0.1)  # 재생이 시작되기를 기다립니다 (값을 조절해야 할 수도 있습니다)

# while player.is_playing():
#     time.sleep(1)  # 재생 중이면 계속 대기



# # 파일 경로 (절대 경로 또는 상대 경로)
# file_path = r"C:\Users\황명주\OneDrive\바탕 화면\Bakjun\falling\Falling-detection\Alarm.mp4"

# # 파일을 기본 프로그램으로 실행
# os.startfile(file_path)

import os
import subprocess

# VLC 실행 경로 (VLC가 설치된 경로에 따라 변경할 것)
vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"

# 파일 경로 (절대 경로 또는 상대 경로)
file_path = r"C:\Users\황명주\OneDrive\바탕 화면\Bakjun\falling\Falling-detection\Alarm.mp4"

# VLC로 오디오만 재생하고 비디오 창을 표시하지 않음
subprocess.Popen([vlc_path, file_path, '--intf', 'dummy', '--dummy-quiet', '--no-video'])

