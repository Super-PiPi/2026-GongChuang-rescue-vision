# ============================================================
# K230 OpenCV HSV 多色色块识别测试
# 在 320x240 图像上检测红、蓝、黄、黑色轮廓，并映射回 800x480 显示。
# 这是独立调参脚本，不加载 YOLO，不发送串口命令。
# ============================================================

import gc
import os
import time

import cv2
from ulab import numpy as np
from media.sensor import Sensor
from media.media import MediaManager
from libs.PipeLine import PipeLine
from libs.Utils import chw2hwc


# ===================== 设备与运行参数 =====================
SENSOR_ID = 1
PIPELINE_SIZE = [800, 480]
CV_SIZE = [320, 240]
DISPLAY_MODE = "lcd"

# 先设为 1 观察完整性能；如果后续与 YOLO 同时运行，可改成 2 或 3。
PROCESS_EVERY_N_FRAMES = 1

# 可以暂时删掉某个名字，只测试需要调节的颜色。
ENABLED_COLORS = ("RED", "BLUE", "YELLOW", "BLACK")

# ===================== HSV 初始阈值 =====================
# OpenCV H:0~179，S/V:0~255。这里只是起始值，必须按赛场光照实机调整。
# 红色横跨 H=0，所以配置两段范围；其余颜色一段即可。
HSV_RANGES = {
    "RED": (
       (np.array([0, 162, 137], dtype=np.uint8), np.array([3, 255, 255], dtype=np.uint8)),
        (np.array([175, 162, 137], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
    ),
    "BLUE": (
        (np.array([34, 20, 49], dtype=np.uint8), np.array([112, 255, 255], dtype=np.uint8)),
    ),
    "YELLOW": (
        (np.array([26, 173, 144], dtype=np.uint8), np.array([34, 255, 255], dtype=np.uint8)),
    ),
    "BLACK": (
        (np.array([0, 0, 0], dtype=np.uint8), np.array([179, 255, 76], dtype=np.uint8)),
    ),
}

OSD_COLORS = {
    "RED": (255, 0, 0),
    "BLUE": (0, 0, 255),
    "YELLOW": (255, 255, 0),
    "BLACK": (255, 255, 255),
}

# ===================== 轮廓过滤参数 =====================
# 全部基于 320x240 的 CV 图像面积，不是 800x480 面积。
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 50000
MIN_BOX_WIDTH = 8
MIN_BOX_HEIGHT = 8
MIN_FILL_RATIO = 0.25
MAX_ASPECT_RATIO = 4.0
MAX_TARGETS_PER_COLOR = 5


def draw_label(img, x, y, text, color=(0, 255, 0), char_size=22):
    img.draw_string_advanced(x + 1, y + 1, char_size, text, (0, 0, 0))
    img.draw_string_advanced(x, y, char_size, text, color)


def build_color_mask(hsv, color_name):
    ranges = HSV_RANGES[color_name]
    mask = None
    for lower, upper in ranges:
        part = cv2.inRange(hsv, lower, upper)
        if mask is None:
            mask = part
        else:
            mask = cv2.bitwise_or(mask, part)
    return mask


def detect_color_blobs(small_rgb, kernel):
    hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
    detections = []

    for color_name in ENABLED_COLORS:
        mask = build_color_mask(hsv, color_name)

        # 开运算去除零碎噪声，闭运算填补物块内部小孔洞。
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color_results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < MIN_BOX_WIDTH or h < MIN_BOX_HEIGHT:
                continue

            box_area = w * h
            if box_area <= 0:
                continue

            fill_ratio = area / box_area
            aspect_ratio = w / h
            if aspect_ratio < 1.0:
                aspect_ratio = 1.0 / aspect_ratio

            if fill_ratio < MIN_FILL_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                continue

            color_results.append({
                "color": color_name,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "cx": int(x + w // 2),
                "cy": int(y + h // 2),
                "area": int(area),
                "fill": fill_ratio,
            })

        color_results.sort(key=lambda item: item["area"], reverse=True)
        detections.extend(color_results[:MAX_TARGETS_PER_COLOR])

        del mask, contours, color_results

    del hsv
    detections.sort(key=lambda item: item["area"], reverse=True)
    return detections


def cv_box_to_source(detection):
    return (
        int(detection["x"] * PIPELINE_SIZE[0] / CV_SIZE[0]),
        int(detection["y"] * PIPELINE_SIZE[1] / CV_SIZE[1]),
        int(detection["w"] * PIPELINE_SIZE[0] / CV_SIZE[0]),
        int(detection["h"] * PIPELINE_SIZE[1] / CV_SIZE[1]),
    )


def source_box_to_display(source_box, display_size):
    x, y, w, h = source_box
    return (
        int(x * display_size[0] / PIPELINE_SIZE[0]),
        int(y * display_size[1] / PIPELINE_SIZE[1]),
        int(w * display_size[0] / PIPELINE_SIZE[0]),
        int(h * display_size[1] / PIPELINE_SIZE[1]),
    )


pl = None
sensor = None

try:
    os.exitpoint(os.EXITPOINT_ENABLE)
    print("===== K230 cv2 color blob test =====")
    print("enabled colors:", ENABLED_COLORS)
    print("Pipeline size:{} CV size:{} process interval:{}".format(
        PIPELINE_SIZE, CV_SIZE, PROCESS_EVERY_N_FRAMES
    ))

    pl = PipeLine(
        rgb888p_size=PIPELINE_SIZE,
        display_mode=DISPLAY_MODE,
        display_size=None,
    )
    sensor = Sensor(id=SENSOR_ID)
    pl.create(sensor)
    display_size = pl.get_display_size()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    frame_count = 0
    last_detections = []
    last_process_ms = 0
    first_shape_printed = False

    while True:
        os.exitpoint()
        frame_chw = pl.get_frame()
        osd_img = pl.osd_img
        osd_img.clear()
        frame_count += 1

        if not first_shape_printed:
            print("PipeLine CHW shape:", frame_chw.shape)

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            start_ms = time.ticks_ms()
            frame_hwc = chw2hwc(frame_chw)
            small_rgb = cv2.resize(
                frame_hwc,
                (CV_SIZE[0], CV_SIZE[1]),
                interpolation=cv2.INTER_AREA,
            )
            last_detections = detect_color_blobs(small_rgb, kernel)
            last_process_ms = time.ticks_diff(time.ticks_ms(), start_ms)

            if not first_shape_printed:
                print("HWC shape:", frame_hwc.shape)
                print("CV image shape:", small_rgb.shape)
                first_shape_printed = True

            del frame_hwc, small_rgb

        color_counts = {"RED": 0, "BLUE": 0, "YELLOW": 0, "BLACK": 0}
        for detection in last_detections:
            color_name = detection["color"]
            color_counts[color_name] += 1
            source_box = cv_box_to_source(detection)
            display_box = source_box_to_display(source_box, display_size)
            osd_color = OSD_COLORS[color_name]
            osd_img.draw_rectangle(display_box, color=osd_color, thickness=2)

            label_x = display_box[0]
            label_y = display_box[1] - 22
            if label_y < 0:
                label_y = display_box[1]
            draw_label(
                osd_img,
                label_x,
                label_y,
                "{} A:{}".format(color_name, detection["area"]),
                osd_color,
                18,
            )

        draw_label(osd_img, 8, 8, "CV BLOB {}ms".format(last_process_ms), (0, 255, 0), 22)
        draw_label(osd_img, 8, 34, "R:{} B:{} Y:{} K:{}".format(
            color_counts["RED"],
            color_counts["BLUE"],
            color_counts["YELLOW"],
            color_counts["BLACK"],
        ), (255, 255, 255), 22)
        pl.show_image()

        if frame_count % 30 == 0:
            print("frame={} process={}ms targets={}".format(
                frame_count, last_process_ms, len(last_detections)
            ))
            for detection in last_detections:
                source_box = cv_box_to_source(detection)
                source_cx = source_box[0] + source_box[2] // 2
                source_cy = source_box[1] + source_box[3] // 2
                print("  {} center=({}, {}) area320={} fill={:.2f}".format(
                    detection["color"],
                    source_cx,
                    source_cy,
                    detection["area"],
                    detection["fill"],
                ))
            gc.collect()

        time.sleep_ms(10)

except KeyboardInterrupt as e:
    print("user stopped:", e)
except BaseException as e:
    print("color blob test FAILED:", e)
finally:
    try:
        if sensor is not None:
            sensor.stop()
    except Exception:
        pass
    try:
        if pl is not None:
            pl.destroy()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    gc.collect()
    print("color blob test finished")

