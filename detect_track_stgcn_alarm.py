import argparse
import time
from pathlib import Path

import os
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# track
from bytetrack_utils.visualize import plot_tracking
from bytetrack_utils.byte_tracker_new import BYTETracker
import numpy as np

# STGCN
from stgcn.ActionsEstLoader import TSSTG

############################수정################################################
icon_img = cv2.imread("sub/icon.png")  # Replace 'icon.png' with your icon file
icon_img = cv2.resize(icon_img, (50, 50))
#################################################################################

##############################################수정된 알고리즘######################################

# send_email.py
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

##################################### WARNING SOUND #############################
import subprocess

# VLC 실행 경로 (VLC가 설치된 경로에 따라 변경할 것)
vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"

# 파일 경로 (절대 경로 또는 상대 경로)
file_path = "sub/Alarm.mp4"
alarm_triggered = False  # 경고음이 시작되었는지 추적하기 위한 플래그

#######################################################################################


# Constants
SMTP_SERVER = 'smtp.naver.com'
SMTP_PORT = 465
ENV_KEY = 'naver_account'
IMAGE_PATH = "sub/gui.jpg"

# Load environment variables
ENV_NAVER = json.loads(os.environ.get('naver_account', '{}'))


class EmailManager:
    def __init__(self):
        self.email_address = None
    
    def set_email(self, email):
        self.email_address = email
    
    def send_email(self):
        if self.email_address:
            result = sendNaver(to=[self.email_address], capture_screen=True)
            print(f"Email sending result: {result}")
        else:
            print("No email address stored.")


# to: 받는 사람 배열
# subject: 메일 제목
# body: 메일 본문
def sendNaver(to=[], subject='[긴급]넘어짐 감지 서비스 알림!!', body='귀하의 보호자께서 넘어진 후 30초간 움직임이 없습니다. 빠르게 조취를 취해주세요.',  capture_screen=False):
    try:
        send_account = 'Naver id address'
        send_pwd = 'pw'
        send_name = 'Your Name'

        smtp = smtplib.SMTP_SSL('smtp.naver.com', 465)
        smtp.login(send_account, send_pwd)

        msg = MIMEMultipart('alternative')

        msg['Subject'] = subject
        msg['From'] = formataddr((str(Header(send_name, 'utf-8')), send_account))
        msg['To'] = ', '.join(to)

        msg.attach(MIMEText(body, 'html'))

        #화면 캡쳐가 필요한 경우
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
        # print("OK")
        return "OK"
    except Exception as ex:
        print('이메일 발송 에러', ex)
        print(send_account)
        print(send_pwd)
        return ex

def gui(email_manager):
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
            email_manager.set_email(email_address)
            print(f"Email address {email_address} has been saved.")
        root.destroy()

    tk.Button(frame, text='확인', command=submit, bg='#0066FF', fg='white', width=10, height=1, font=('Arial', 14)).pack()
    root.mainloop()
##########################################################################################################################################

def detect(opt):
    source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label, track_thresh, track_buffer, match_thresh =\
        opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label,opt.track_thresh,opt.track_buffer,opt.match_thresh
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    fall_down_count = 0
    fps = 30  # 가정: 비디오의 프레임 레이트가 30fps
    threshold = 2 * fps  # 1분 동안 지속되어야 함

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    # Tracker
    tracker = BYTETracker(track_thresh, track_buffer, match_thresh)  # ct+++
    frame_id = 0  # ct+++

    # Actions Estimate.
    action_model = TSSTG()

    if isinstance(imgsz, (list, tuple)):
        assert len(imgsz) == 2;  "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

        if frame_idx % 1 != 0:
            continue

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # print(pred[...,4].max())
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # det contains 51 key point results (x, y, cof) and 6 box results (xyxy, cof, cla).
            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                Results = []  # ct+++
                Result_kpts = []

                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    Results.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf)])  # ct+++
                    # print(np.array(Results))

                    # [[x,y,c],[x,y,c]...]
                    Result_kpts.append(del_tensor_ele(det[det_index, 6:], 3, 15).reshape(-1, 3))  # 存储13个kpt
                # print(Result_kpts)

                online_tlwhs, online_ids, action_results, action_name = track_main(tracker, np.array(Results), Result_kpts, frame_id,
                                                                      1080, 1920,
                                                                        (1080, 1920), action_model, im0)  # ct+++
                online_tlwhs = np.array(online_tlwhs).reshape(-1, 4)
                online_xyxys = tlwh2xyxy(online_tlwhs)
                outputs = np.concatenate((online_xyxys, np.array(online_ids).reshape(-1, 1)), axis=1)
                
                # Write results
                for det_index, (output, conf) in enumerate(zip(outputs, det[:, 4])):

                    xyxy = output[0:4]
                    id = output[4]
                    #print(xyxy)

                    if save_txt:
                        # to MOT format
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2] - output[0]
                        bbox_h = output[3] - output[1]
                        # Write MOT compliant results to file
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                    if view_img:  # Add bbox to image
                        label = f'{int(id)} {conf:.2f} {str(action_results[det_index])}'
                        kpts = det[det_index, 6:]
                        warning_color = action_name
##################################################수정################################################

                        if action_name=="Fall Down":
                            plot_one_box(xyxy, im0, label=label, color=(84, 61, 247),
                                     line_thickness=opt.line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3,
                                     orig_shape=im0.shape[:2], line_color2=(84, 61, 247), thickness2=16)
                            # Calculate the center coordinates of the bounding box
                            x_center = int((xyxy[0] + xyxy[2]) / 2)
                            y_center = int((xyxy[1] + xyxy[3]) / 2)

                             # Calculate the top-left coordinates to place the icon
                            x1 = x_center - 25  # Half of the resized icon's width
                            y1 = y_center - 25  # Half of the resized icon's height
                            x2 = x1 + 50
                            y2 = y1 + 50

                            
                            # Overlay the resized icon at the center of the bounding box
                            overlay_image(im0, icon_img, x1, y1, x2, y2)
                        else:
                            plot_one_box(xyxy, im0, label=label, color=None,
                                     line_thickness=opt.line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3,
                                     orig_shape=im0.shape[:2])
##########################################################################################################################
                        
##############################################수정된 알고리즘######################################
                
                if action_name == 'Fall Down':
                    fall_down_count += 1
                    # 넘어짐 감지 횟수가 임계값에 도달하고, 경고음이 아직 시작되지 않았다면
                    if fall_down_count >= threshold and not alarm_triggered:
                        # 경고음 시작
                        subprocess.Popen([vlc_path, file_path, '--intf', 'dummy', '--dummy-quiet', '--no-video'])
                        # 이메일 전송
                        email_manager.send_email()
                        print("경고: 사용자가 넘어진지 30초 동안 움직임이 없습니다!")
                        # 경고음 시작 플래그 설정
                        alarm_triggered = True
                        # 카운터 초기화 (경고음이 한 번만 실행되도록 하려면 이 부분을 제거합니다)
                        # fall_down_count = 0
                else:
                    # 'Fall Down'이 아니면 카운터와 플래그 초기화
                    fall_down_count = 0
                    alarm_triggered = False
                
##################################################################################################
                    

                        

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')
            
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
                
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)



# ct+++   action_model, frame
def track_main(tracker, detection_results, Result_kpts, frame_id, image_height, image_width, test_size,
               action_model, frame):
    '''
    main function for tracking
    :param args: the input arguments, mainly about track_thresh, track_buffer, match_thresh
    :param detection_results: the detection bounds results, a list of [x1, y1, x2, y2, score]
    :param frame_id: the current frame id
    :param image_height: the height of the image
    :param image_width: the width of the image
    :param test_size: the size of the inference model
    '''
    online_targets = tracker.update(detection_results, Result_kpts, [image_height, image_width], test_size)
    online_tlwhs = []
    online_ids = []
    online_scores = []
    results = []
    aspect_ratio_thresh = 1.6  # +++++
    min_box_area = 10  # ++++
    action_results = []
    action_name = 'pending..'

    for target in online_targets:
        tlwh = target.tlwh
        tid = target.track_id
        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > min_box_area or vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(target.score)
            # save results
            results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{target.score:.2f},-1,-1,-1\n"
                    )
        
        #######################수정된 알고리즘######################################
        # print(results)
        action = 'pending..'
        clr = (0, 255, 0)
        # Use 30 frames time-steps to prediction.
        if len(target.keypoints_list) == 30:
            pts = np.array(target.keypoints_list, dtype=np.float32)
            out = action_model.predict(pts, frame.shape[:2])
            action_name = action_model.class_names[out[0].argmax()]
            if action_name != 'Fall Down' :
                if tlwh[2] >= (tlwh[3]*1.5) :
                    action_name = 'Fall Down'
            action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
            if action_name == 'Fall Down':
                clr = (255, 0, 0)
            elif action_name == 'Lying Down':
                clr = (255, 200, 0)
            action = action
        action_results.append(action)
        ############################################################################

    return online_tlwhs, online_ids, action_results, action_name  # ct+++action_results IADD action_name


def tlwh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 0] + x[:, 2]  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3]  # bottom right y
    return y


# kpts = del_tensor_ele(kpts, 3, 15)
def del_tensor_ele(arr, index_a, index_b):
    arr1 = arr[0:index_a]
    arr2 = arr[index_b:]
    return torch.cat((arr1, arr2), dim=0)

#######################수정된 알고리즘######################################
def overlay_image(background, overlay, x1, y1, x2, y2):
    # Resize overlay image to 50x50 pixels
    overlay_resized = cv2.resize(overlay, (50, 50))
    
    # Place the resized overlay image onto the background frame
    background[y1:y2, x1:x2] = cv2.addWeighted(background[y1:y2, x1:x2], 1, overlay_resized, 1, 0)
############################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', nargs= '+', type=int, default=960, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=True, action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', default=True, action='store_true', help='use keypoint labels')

    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")  # ct+++
    parser.add_argument("--track_buffer", type=int, default=30,
                        help="the frames for keep lost tracks, usually as same with FPS")  # ct+++
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")  # ct+++

    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            email_manager = EmailManager()
            gui(email_manager)
            detect(opt=opt)
