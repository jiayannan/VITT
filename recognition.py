import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import tqdm
import copy

from cam_utils import CamUtils
from face_utils import FaceUtils

import threading
import queue

from utils.timer import Timer
from utils.log import get_logger
from utils.mjpeg import MjpegServer
from tracker.multitracker import JDETracker
from utils.datasets import letterbox
from utils import visualization as vis
from utils.parse_config import parse_model_cfg


class VideoTracker(object):
    def __init__(self, args, video_path):
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.rtsp != "":
            print("Using rtsp " + str(args.rtsp))
            self.vdo = cv2.VideoCapture(args.rtsp)
        elif args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()

        frame_rate = 30
        cfg_dict = parse_model_cfg(args.cfg)
        self.width, self.height = int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])
        args.img_size = [int(cfg_dict[0]['width']), int(cfg_dict[0]['height'])]
        self.tracker = JDETracker(args, frame_rate=frame_rate)

        self.known_faces = {}

        self.faceUtils = FaceUtils()
        self.camUtils = CamUtils(args)
        self.tmp_imgs = []
        self.tmp_moves = []
        self.result_imgs = queue.Queue()
        self.is_running = True
        print("Loading Control Done.")


    def __enter__(self):
        if self.args.rtsp != "":
            ret, frame = self.vdo.read()
            assert ret, "Error: RTSP error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]
        elif self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]
        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            fps = int(self.vdo.get(cv2.CAP_PROP_FPS))
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, fps, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def read_thread(self):
        while True:
            time_start = time.time()
            # _, ori_im = self.vdo.retrieve()
            ret, ori_im = self.vdo.read()
            if not ret:
                if self.args.rtsp != "":
                    print("Using rtsp " + str(self.args.rtsp))
                    self.vdo = cv2.VideoCapture(self.args.rtsp)
                elif self.args.cam != -1:
                    print("Using webcam " + str(self.args.cam))
                    self.vdo = cv2.VideoCapture(self.args.cam)
                else:
                    self.vdo = cv2.VideoCapture()
                continue

            # ori_im = cv2.resize(ori_im, (800, 600))
            self.tmp_imgs.append(copy.deepcopy(ori_im))

            # print("read time: {}".format(time.time() - time_start))

    def jpeg_thread(self):
        mjpeg_server = MjpegServer(port=8080)
        print('MJPEG server started... 8080')
        while self.is_running:
            ori_im = self.result_imgs.get()
            mjpeg_server.send_img(ori_im)
        mjpeg_server.shutdown()

    def move_thread(self):
        while True:
            if len(self.tmp_moves) == 0:
                continue
            center, xywh = copy.deepcopy(self.tmp_moves[-1])
            self.tmp_moves = []
            self.camUtils.move_cam(center, xywh)

    def start_thread(self):
        thread_read = threading.Thread(target=self.read_thread, args=())
        thread_read.daemon = True
        thread_read.start()

        thread_jpeg = threading.Thread(target=self.jpeg_thread, args=())
        thread_jpeg.daemon = True
        thread_jpeg.start()

        thread_move = threading.Thread(target=self.move_thread, args=())
        thread_move.daemon = True
        thread_move.start()


    def run(self):
        results = []
        idx_frame = 0
        idx_frame1 = 0
        face_names = {}

        self.start_thread()

        timer = Timer()

        while True:
            # _, ori_im = self.vdo.retrieve()
            if self.args.rtsp != "" or self.args.cam != -1:
                if len(self.tmp_imgs) == 0:
                    continue
                ori_im = copy.deepcopy(self.tmp_imgs[-1])
                self.tmp_imgs = []
            else:
                ret, ori_im = self.vdo.read()

            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue
            idx_frame1 += 1

            start = time.time()

            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            center = (int(ori_im.shape[1]/2), int(ori_im.shape[0]/2))
            # cv2.circle(ori_im, center, 2, (255,0,0), 0)

            img, _, _, _ = letterbox(ori_im, height=self.height, width=self.width)
            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0

            # run tracking
            timer.tic()

            time_start = time.time()
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
            print("input time: {}".format(time.time() - time_start))

            time_start = time.time()
            online_targets = self.tracker.update(blob, ori_im)
            print("tracker time: {}".format(time.time() - time_start))

            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            timer.toc()

            for tlwh, tid in zip(online_tlwhs, online_ids):
                face_name = face_names.get(tid)
                if face_name is None:
                    bb_xyxy = tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
                    face_name = self.faceUtils.detect_face(im, bb_xyxy)
                    face_names[tid] = face_name
                print("{} {}".format(tid, face_names.get(tid)))

                if (idx_frame1 % self.args.move_skip) == 0 and face_names.get(tid) == self.args.obj:
                    print("move")
                    # self.tmp_moves.append((center, xywh))
                    self.camUtils.move_cam(center, tlwh)

            # save results
            results.append((idx_frame - 1, online_tlwhs, online_ids))

            ori_im = vis.plot_tracking(ori_im, online_tlwhs, online_ids, frame_id=idx_frame,
                                          fps=1. / timer.average_time)
            end = time.time()

            self.result_imgs.put_nowait(ori_im)

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), len(online_tlwhs), len(online_ids)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=3)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument("--rtsp", action="store", dest="rtsp", type=str, default="")
    parser.add_argument("--obj", type=str, default="")

    parser.add_argument("--onvif_host", type=str, default="192.168.199.240")
    parser.add_argument("--onvif_port", type=int, default=80)
    parser.add_argument("--onvif_user", type=str, default="admin")
    parser.add_argument("--onvif_pswd", type=str, default="Ab12345678")

    parser.add_argument("--move_wait", type=float, default=0.05)
    parser.add_argument("--move_scale", type=float, default=0.9)
    parser.add_argument("--move_skip", type=int, default=1)
    parser.add_argument("--move_flip", type=int, default=0)

    parser.add_argument("--type", type=str, default="det", choices=("reg", "det", "get"))

    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, default="yolov4-608",
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))

    parser.add_argument('--cfg', type=str, default='cfg/yolov3_1088x608.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
    parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
    parser.add_argument('--track-buffer', type=int, default=30, help='tracking buffer')
    parser.add_argument('--input-video', type=str, help='path to the input video')
    parser.add_argument('--output-format', type=str, default='video', choices=['video', 'text'], help='Expected output format. Video or text.')
    parser.add_argument('--output-root', type=str, default='results', help='expected output root path')

    return parser.parse_args()

def main():
    args = parse_args()

    names = [args.obj]
    if args.type == "reg": # 注册人脸
        face(args.VIDEO_PATH, "face_{}".format(names[0]))
        return
    elif args.type == "get": # 获取RTSP地址
        CamUtils(args)
        return

    with VideoTracker(args, video_path=args.VIDEO_PATH) as vdo_trk:
        # vdo_trk.register_faces("faces", "chenjinju")
        # for name in ["lizhongyi", "liyuanyuan", "zhuxiaoning"]:
        #     vdo_trk.register_faces("face_"+name, name)
        # for name in ["p1", "p2", "p3", "p4", "p5"]:
        #     vdo_trk.register_faces("face_" + name, name)
        for name in names:
            print("register face {}".format(name))
            vdo_trk.faceUtils.register_faces("face_" + name, name)
        for name, features in vdo_trk.known_faces.items():
            print("{} {}".format(name, len(features)))

        vdo_trk.run()

def face(video_path, feature_path):
    print("register face")
    faceRec = FaceUtils()
    faceRec.rec_video(video_path, feature_path)

def control():
    args = parse_args()
    con = CamUtils(args)
    # for i in range(10):
    #     con.move_up(0.05, 0.9)
    con.move(1,1)

if __name__ == "__main__":
    # face("zhuxiaoning.mp4", "face_zhuxiaoning")
    main()
    # control()

