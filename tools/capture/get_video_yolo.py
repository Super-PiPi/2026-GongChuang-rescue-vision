# 自动采图 + 屏幕实时画框 (极客降维打击版)
# K230 CanMV
# 完全使用最稳定的 PipeLine 跑模型，彻底放弃导致崩溃的底层图片转换
# 直接将底层张量以 .bin 保存，回电脑转换，完美避开所有 Bug！

import os
import time
import gc

from media.sensor import *
from media.display import *
from media.media import *
from libs.PipeLine import PipeLine
from libs.YOLO import YOLOv5

# =========================
# 参数区
# =========================
CAMERA_ID = 1
IMG_WIDTH = 800
IMG_HEIGHT = 480

SAVE_DIR = "/data/dataset"
IMG_PREFIX = "img_0720_"
IMG_EXT = ".bin"   # 【核心改变】保存为二进制阵列文件
NUM_WIDTH = 5

CAPTURE_PER_SECOND = 1
CAPTURE_INTERVAL_MS = int(1000 / CAPTURE_PER_SECOND)

KMODEL_PATH = "/sdcard/model/best_0719.kmodel"
MODEL_INPUT_SIZE = [320, 320]
LABELS = [
    "redball", "blueball", "redboard", "blueboard", "redsection",
    "bluesection", "shizi", "redcube", "bluecube", "greenzhu",
    "yellowtai", "blackball", "yellowball",
]
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45
MAX_BOXES_NUM = 50

def ensure_dir(path):
    parts = path.split("/")
    cur = ""
    for p in parts:
        if p == "": continue
        cur += "/" + p
        try: os.stat(cur)
        except: os.mkdir(cur)

def get_next_index(save_dir):
    max_id = 0
    try:
        files = os.listdir(save_dir)
        for name in files:
            if name.startswith(IMG_PREFIX) and name.endswith(IMG_EXT):
                try:
                    num = int(name[len(IMG_PREFIX):-len(IMG_EXT)])
                    if num > max_id: max_id = num
                except: pass
    except: pass
    return max_id + 1

def get_display_label(label):
    name_map = {
        "redball": "RED", "blueball": "BLUE", "blackball": "CORE",
        "yellowball": "DANGER", "redboard": "REDBOARD", "blueboard": "BLUEBOARD",
    }
    return name_map.get(label, str(label))

def parse_yolo_results(res):
    dets = []
    if res is None: return dets
    def append_det(x1, y1, x2, y2, score, class_id, label):
        if label is None and class_id is not None:
            try:
                if 0 <= int(class_id) < len(LABELS): label = LABELS[int(class_id)]
            except: label = None
        try:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if x2 < x1 or y2 < y1:
                x2, y2 = x1 + max(0, x2), y1 + max(0, y2)
            w, h = max(0, x2 - x1), max(0, y2 - y1)
            if w == 0 or h == 0: return
            dets.append({
                "x1": x1, "y1": y1, "rect": (x1, y1, w, h),
                "cx": x1 + w // 2, "cy": y1 + h // 2,
                "score": float(score) if score else 0.0, "label": label
            })
        except: pass

    if isinstance(res, (tuple, list)) and len(res) == 3:
        boxes, class_ids, scores = res[0], res[1], res[2]
        if isinstance(boxes, (tuple, list)) and isinstance(class_ids, (tuple, list)):
            for i in range(min(len(boxes), len(class_ids))):
                if boxes[i] and len(boxes[i]) >= 4:
                    s = scores[i] if isinstance(scores, (tuple, list)) else 0.0
                    append_det(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], s, class_ids[i], None)
            return dets

    for entry in res:
        if not entry: continue
        if isinstance(entry, dict):
            x = entry.get("x1", entry.get("x", 0))
            y = entry.get("y1", entry.get("y", 0))
            w = entry.get("w", 0)
            h = entry.get("h", 0)
            s = entry.get("score", entry.get("confidence", 0.0))
            c = entry.get("class_id", entry.get("classid"))
            l = entry.get("label")
            append_det(x, y, x + w, y + h, s, c, l)
    return dets

def draw_yolo_detections(img, detections):
    for det in detections:
        label_str = str(det["label"])
        if "red" in label_str: color = (255, 0, 0)
        elif "blue" in label_str: color = (0, 0, 255)
        elif "yellow" in label_str: color = (255, 255, 0)
        elif "black" in label_str: color = (255, 255, 255)
        else: color = (0, 255, 0)

        img.draw_rectangle(det["rect"], color=color, thickness=2)
        img.draw_cross(det["cx"], det["cy"], color=color, size=8)
        text = "{} {:.2f}".format(get_display_label(det["label"]), det["score"])
        img.draw_string_advanced(det["x1"], max(0, det["y1"] - 24), 24, text, color)

def main():
    print(">>> 启动极客降维打击采图模式...")
    os.exitpoint(os.EXITPOINT_ENABLE)

    ensure_dir(SAVE_DIR)
    next_id = get_next_index(SAVE_DIR)
    print(">>> 下一个文件编号 =", next_id)

    # 1. 采用最稳如老狗的 PipeLine 架构
    pl = PipeLine(rgb888p_size=[IMG_WIDTH, IMG_HEIGHT], display_mode="lcd", display_size=[IMG_WIDTH, IMG_HEIGHT])
    sensor = Sensor(id=CAMERA_ID)
    pl.create(sensor)
    display_size = pl.get_display_size()

    # 2. YOLO 使用 video 模式，完美榨干硬件加速
    yolo = YOLOv5(
        task_type="detect", mode="video", kmodel_path=KMODEL_PATH,
        labels=LABELS, rgb888p_size=[IMG_WIDTH, IMG_HEIGHT],
        model_input_size=MODEL_INPUT_SIZE, display_size=display_size,
        conf_thresh=CONFIDENCE_THRESHOLD, nms_thresh=NMS_THRESHOLD,
        max_boxes_num=MAX_BOXES_NUM, debug_mode=0,
    )
    yolo.config_preprocess()

    last_capture_ms = time.ticks_ms()
    saved_count = 0

    try:
        while True:
            os.exitpoint()
            # 拿到最底层的张量指针
            img = pl.get_frame()

            osd_img = pl.osd_img
            try: osd_img.clear()
            except: pass

            now_ms = time.ticks_ms()

            # --- 步骤 A：直接把底层张量暴力 Dump 进 SD 卡 ---
            if time.ticks_diff(now_ms, last_capture_ms) >= CAPTURE_INTERVAL_MS:
                last_capture_ms = now_ms
                file_name = "{}{:05d}{}".format(IMG_PREFIX, next_id, IMG_EXT)
                file_path = "{}/{}".format(SAVE_DIR, file_name)

                try:
                    # 极速写入二进制文件，绝对不触发任何格式转换崩溃！
                    with open(file_path, "wb") as f:
                        try: f.write(img)
                        except Exception: f.write(img.tobytes())

                    saved_count += 1
                    next_id += 1
                    print("Saved RAW Data:", file_path)
                except Exception as e:
                    print("Save Error:", e)

            # --- 步骤 B：正常跑 YOLO 并在悬浮板上画框 ---
            try:
                res = yolo.run(img)
                dets = parse_yolo_results(res)
                draw_yolo_detections(osd_img, dets)
            except Exception: pass

            osd_img.draw_string_advanced(10, 10, 30, "[Recording] Saved: {} (.bin)".format(saved_count), (0, 255, 0))

            pl.show_image()
            gc.collect()

    except KeyboardInterrupt:
        print("user stop")
    except BaseException as e:
        print("run error:", e)
    finally:
        print("stopping...")
        try:
            if sensor: sensor.stop()
        except: pass
        try:
            if yolo is not None: yolo.deinit()
        except: pass
        try:
            if pl is not None: pl.destroy()
        except: pass
        try: MediaManager.deinit()
        except: pass

if __name__ == "__main__":
    main()
