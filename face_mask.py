#!/usr/bin/env python
# coding: utf-8

# ### Import library

# In[1]:


from argparse import ArgumentParser

from responsive_voice.voices import UKEnglishMale

from inference import FaceDetection, MaskDetection
from pyvino_utils import InputFeeder
from IPython import display

import cv2
import time
import numpy as np
from loguru import logger
import time

from flask import Flask, render_template, Response

import sys

# ### Set the arguments
class set_args():
    face_model = "models/face-detection-adas-0001"
    mask_model = "models/face_mask"
    input = "cam"
    device = "GPU"
    face_prob_threshold = 0.8
    mask_prob_threshold = 0.3
    enable_speech = False
    tts = "Please wear your MASK!!"; # Text-to-Speech, used for notification.
    ffmpeg = False; # Flush video to FFMPEG
    show_bbox = True; # Show bounding box and stats on screen [debugging].
    debug = True; # Show output on screen [debugging].
    width = 640
    height = 480

def arg_parser():
    """Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-f",
        "--face-model",
        required=False,
        default="models/face-detection-adas-0001",
        type=str,
        help="Path to an xml file with a trained model.",
    )
    parser.add_argument(
        "-m",
        "--mask-model",
        required=False,
        default="models/face_mask",
        type=str,
        help="Path to an xml file with a trained model.",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=False,
        default="cam",
        type=str,
        help="Path to image or video file or 'cam' for Webcam.",
    )

    parser.add_argument(
        "-d",
        "--device",
        default="GPU",
        help="Specify the target device to infer on: "
        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
        "will look for a suitable plugin for device "
        "specified (GPU by default)",
    )
    parser.add_argument(
        "--face_prob_threshold",
        type=float,
        default=0.8,
        help="Probability threshold for face detections filtering (Default: 0.8)",
    )
    parser.add_argument(
        "--mask_prob_threshold",
        type=float,
        default=0.3,
        help="Probability threshold for face mask detections filtering" "(Default: 0.3)",
    )
    parser.add_argument(
        "--enable-speech", action="store_true", help="Enable speech notification.",
    )
    parser.add_argument(
        "--tts",
        type=str,
        default="Please wear your MASK!!",
        help="Text-to-Speech, used for notification.",
    )
    parser.add_argument(
        "--ffmpeg", action="store_true", help="Flush video to FFMPEG.",
    )
    parser.add_argument(
        "--show-bbox",
        action="store_true",
        help="Show bounding box and stats on screen [debugging].",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show output on screen [debugging].",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Input Width (Default: 640)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Input Height (Default: 480)",
    )

    return parser.parse_args()
    
def gen_frames_raw():
    for frame in input_feed.next_frame(progress=False, stabilize_video=False):
        # encode numpy array to jpg
        _, encoded_img = cv2.imencode(
            ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
        )
        bytes_frame = encoded_img.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytes_frame + b'\r\n')  # concat frame one by one

def gen_frames_inference():       
    count = 0
    face_detect_infer_time = 0
    mask_detect_infer_time = 0
    mask_detected_prob = -1
    try:
        # TODO: Convert to contextmanager
        face_detect_infer_time_samples = []
        mask_detect_infer_time_samples = []
        for frame in input_feed.next_frame():
            count += 1

            start_time = time.time()
            fd_results = face_detection.predict(
                frame, show_bbox=args.show_bbox, mask_detected=mask_detected_prob
            )
            face_detect_infer_time_samples.append(time.time() - start_time)
            face_bboxes = fd_results["process_output"]["bbox_coord"]
            if face_bboxes:
                for face_bbox in face_bboxes:
                    # Useful resource:
                    # https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

                    # Face bounding box coordinates cropped from the face detection
                    # inference are face_bboxes i.e `xmin, ymin, xmax, ymax`
                    # Therefore the face can be cropped by:
                    # frame[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]

                    # extract the face ROI
                    (x, y, w, h) = face_bbox
                    face = frame[y:h, x:w]
                    (face_height, face_width) = face.shape[:2]
                    # Crop and show face
                    # input_feed.show(frame[y:h, x:w], "face")

                    # ensure the face width and height are sufficiently large
                    if face_height < 20 or face_width < 20:
                        continue

                    start_time = time.time()
                    md_results = mask_detection.predict(
                        face, show_bbox=args.show_bbox, frame=frame
                    )
                    mask_detect_infer_time_samples.append(time.time() - start_time)
                    mask_detected_prob = md_results["process_output"][
                        "flattened_predictions"
                    ]
                    if (
                        int(count) % 200 == 1
                        and args.enable_speech
                        and float(mask_detected_prob) < args.mask_prob_threshold
                    ):
                        engine.play_mp3(speak)

            # Calculate the avg inferecing time every 200 samples
            if int(count) % 5 == 1:
                if len(face_detect_infer_time_samples):
                    face_detect_infer_time = np.mean(face_detect_infer_time_samples) * 1000
                    face_detect_infer_time_samples = []
                else:
                    face_detect_infer_time = 0
                if len(mask_detect_infer_time_samples):                        
                    mask_detect_infer_time = np.mean(mask_detect_infer_time_samples) * 1000
                    mask_detect_infer_time_samples = []
                else:
                    mask_detect_infer_time = 0

            if args.debug:
                text = f"Face Detection Inference time: {face_detect_infer_time:.3f} ms"
                input_feed.add_text(text, frame, (15, input_feed.source_height - 80))
                text = (
                    f"Face Mask Detection Inference time: {mask_detect_infer_time:.3f} ms"
                )
                input_feed.add_text(text, frame, (15, input_feed.source_height - 60))

            # input_feed.show(input_feed.resize(frame))
            frame = input_feed.resize(frame)
            _, encoded_img = cv2.imencode(
                ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
            )
            bytes_frame = encoded_img.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + bytes_frame + b'\r\n')  # concat frame one by one
            '''
            i = display.Image(data=encoded_img)
            # Display the image in this notebook
            display.clear_output(wait=True)
            display.display(i)
            '''

    finally:
        # input_feed.close()
        pass
        
# Parse arguments
args = arg_parser()
       
# Initialize the video stream
# logger.info(f"Loaded input source type: {self._input_type}")
input_feed = InputFeeder(input_feed=args.input)
input_feed.resize_cam_input(args.height, args.width)

if args.enable_speech:
    # TODO: Add args for selecting language, accent and male/female voice
    engine = UKEnglishMale()
    speak = engine.get_mp3(args.tts)

logger.info("Loading face detection model")
start_time = time.time()
face_detection = FaceDetection(
    model_name=args.face_model,
    device=args.device,
    threshold=args.face_prob_threshold,
    input_feed=input_feed,
)
logger.info(f"Elapsed time: {(time.time() - start_time)/1.0:.2f} secs")

logger.info("Loading mask detection model")
start_time = time.time()
mask_detection = MaskDetection(
    model_name=args.mask_model,
    device=args.device,
    threshold=args.mask_prob_threshold,
)       
logger.info(f"Elapsed time: {(time.time() - start_time)/1.0:.2f} secs")
        
# Start the Web service
app = Flask(__name__, template_folder='./templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames_inference(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the web service
app.run(debug=False, host="0.0.0.0")
