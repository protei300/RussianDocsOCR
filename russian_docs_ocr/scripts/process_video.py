# -*- coding: UTF-8 -*-
import sys
sys.path.append('..')
import cv2
import os
import time
from document_processing import Pipeline
import argparse
import warnings

sys.path.append('..')
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DISPLAY_IS_AVAILABLE = os.getenv('DISPLAY') is not None or os.name != 'posix'


if __name__ == '__main__':
    '''
    Video processing script shows how to work with video streams or webcams.
    To run script use the command line
    python process_video.py
    - with default parameters
    python process_video.py -v 'http://192.168.0.1:8080/video' -z 720p -d gpu
    or just run this script in IDE
    For using GPU make sure that CUDA execution provider has been installed properly
    Installation guide for Windows and for Linux is here
    https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
    '''
    parser = argparse.ArgumentParser(description='Video processing')

    parser.add_argument('-v', '--video_url',
                        help='Choose video url or webcam. You can use file name like test_video.mov. You can use '
                             '"webcam" or just a file name like test_video.mov or video stream like '
                             'http://192.168.0.1:8080/video', type=str,
                        default='webcam')

    parser.add_argument('-z', '--screen_size',
                        help='Select a screen size. In demo you can choose only 720p (1440, 720) or 1080p (1920, 1080)', type=str,
                        default='720p')

    parser.add_argument('-d', '--device', help='On which device to run - cpu or gpu', default='gpu', type=str)

    args = parser.parse_args()
    params = vars(args)

    print(f'Start proccessing {params["video_url"]} with screen size {str(params["screen_size"])} on {str(params["device"])}...')

    # Choosing the video source
    if params['video_url'] == 'webcam':
        cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    else:
        cap = cv2.VideoCapture(params['video_url'], cv2.CAP_ANY)

    # Choosing the resolution
    if params['screen_size'] == '1080p':
        cap.set(3, 1920)
        cap.set(4, 1080)
    # default screen size is 720p
    else:
        cap.set(3, 1440)
        cap.set(4, 720)

    # pipeline initialisation
    pipeline = Pipeline(model_format='ONNX', device=params['device'], )

    # starting performing frames and calculating statistics
    frames = 0
    fps = 0
    frame_time = time.time()
    frame_time_in_sec = 0
    while True:

        frame_time = time.time() - frame_time
        frame_time_in_sec = frame_time_in_sec + frame_time
        frames += 1
        if frame_time_in_sec > 1:
            frame_time_in_sec = 0
            fps = frames
            frames = 0

        if fps != 0:
            print(f'fps = {fps} Frame performing time = {str(int(frame_time * 1000))}ms')

        frame_time = time.time()

        ret, img = cap.read()

        # check the existence of a video source
        if img is not None:
            original_image = img.copy()
        else:
            print('Camera is not connected or video stream inaccessible. Check camera connection or a video stream.')
            cap.release()
            cv2.destroyAllWindows()
            break

        cv2.putText(img, 'FPS = ' + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (260, 80, 80), 1)
        ###
        result = pipeline(original_image, check_quality=False, low_quality=True, docconf=0.2, img_size=1500)
        ocr_result = result.ocr
        print(ocr_result)
        ###
        if DISPLAY_IS_AVAILABLE:
            cv2.imshow("Camera", img)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        current_time = time.time()

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
