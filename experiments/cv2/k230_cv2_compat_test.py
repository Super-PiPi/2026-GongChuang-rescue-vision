# ============================================================
# K230 OpenCV 实机兼容性验证
# 验证 cv2 API、PipeLine 的 CHW->HWC 转换以及 320x240 处理耗时。
# 不加载 YOLO，不发送串口命令，不修改任何 V8.x 程序。
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


SENSOR_ID = 1
PIPELINE_SIZE = [800, 480]
CV_SIZE = [320, 240]
DISPLAY_MODE = "lcd"

# 实拍阶段仅用红色做通路验证。红色在 HSV 中跨越 0 度，需要两段阈值。
RED_LOW_1 = np.array([0, 80, 50], dtype=np.uint8)
RED_HIGH_1 = np.array([12, 255, 255], dtype=np.uint8)
RED_LOW_2 = np.array([165, 80, 50], dtype=np.uint8)
RED_HIGH_2 = np.array([179, 255, 255], dtype=np.uint8)


def draw_label(img, x, y, text, color=(0, 255, 0)):
    img.draw_string_advanced(x + 1, y + 1, 24, text, (0, 0, 0))
    img.draw_string_advanced(x, y, 24, text, color)


def check_required_apis():
    required = (
        "resize",
        "cvtColor",
        "inRange",
        "bitwise_or",
        "morphologyEx",
        "getStructuringElement",
        "findContours",
        "contourArea",
        "boundingRect",
        "rectangle",
    )
    missing = []
    for name in required:
        if not hasattr(cv2, name):
            missing.append(name)

    print("cv2 module imported successfully")
    if hasattr(cv2, "__version__"):
        print("cv2 version:", cv2.__version__)
    print("required APIs missing:", missing)

    if missing:
        raise RuntimeError("current firmware lacks required cv2 APIs: {}".format(missing))


def synthetic_self_test():
    """不用摄像头构造一个红色矩形，验证完整 OpenCV 调用链。"""
    test_rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    cv2.rectangle(test_rgb, (20, 15), (79, 64), (255, 0, 0), -1)

    hsv = cv2.cvtColor(test_rgb, cv2.COLOR_RGB2HSV)
    mask_1 = cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1)
    mask_2 = cv2.inRange(hsv, RED_LOW_2, RED_HIGH_2)
    mask = cv2.bitwise_or(mask_1, mask_2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 1:
        raise RuntimeError("synthetic test expected 1 contour, got {}".format(len(contours)))

    area = cv2.contourArea(contours[0])
    rect = cv2.boundingRect(contours[0])
    print("synthetic test PASS: area={} rect={}".format(area, rect))


def free_memory_text():
    try:
        return str(gc.mem_free())
    except Exception:
        return "N/A"


pl = None
sensor = None

try:
    os.exitpoint(os.EXITPOINT_ENABLE)
    print("===== K230 cv2 compatibility test =====")
    check_required_apis()
    synthetic_self_test()
    gc.collect()
    print("free memory before camera:", free_memory_text())

    # 与 vision_V8.6.py 相同：PipeLine 输出 800x480 的 RGBP888/CHW ndarray。
    pl = PipeLine(
        rgb888p_size=PIPELINE_SIZE,
        display_mode=DISPLAY_MODE,
        display_size=None,
    )
    sensor = Sensor(id=SENSOR_ID)
    pl.create(sensor)
    display_size = pl.get_display_size()

    frame_count = 0
    total_convert_ms = 0
    total_cv_ms = 0
    first_shape_printed = False
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    while True:
        os.exitpoint()
        frame_chw = pl.get_frame()
        osd_img = pl.osd_img
        osd_img.clear()

        if not first_shape_printed:
            print("PipeLine CHW shape:", frame_chw.shape)

        t0 = time.ticks_ms()
        frame_hwc = chw2hwc(frame_chw)
        t1 = time.ticks_ms()

        # 先完成 CHW->HWC，再缩小；这正是未来接入当前主程序时需要验证的路径。
        small_rgb = cv2.resize(frame_hwc, (CV_SIZE[0], CV_SIZE[1]), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
        mask_1 = cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1)
        mask_2 = cv2.inRange(hsv, RED_LOW_2, RED_HIGH_2)
        mask = cv2.bitwise_or(mask_1, mask_2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t2 = time.ticks_ms()

        convert_ms = time.ticks_diff(t1, t0)
        cv_ms = time.ticks_diff(t2, t1)
        total_convert_ms += convert_ms
        total_cv_ms += cv_ms
        frame_count += 1

        valid_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100 or area > 50000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            # 从 320x240 CV 坐标映射回 800x480，再映射到实际显示尺寸。
            src_x = int(x * PIPELINE_SIZE[0] / CV_SIZE[0])
            src_y = int(y * PIPELINE_SIZE[1] / CV_SIZE[1])
            src_w = int(w * PIPELINE_SIZE[0] / CV_SIZE[0])
            src_h = int(h * PIPELINE_SIZE[1] / CV_SIZE[1])
            draw_x = int(src_x * display_size[0] / PIPELINE_SIZE[0])
            draw_y = int(src_y * display_size[1] / PIPELINE_SIZE[1])
            draw_w = int(src_w * display_size[0] / PIPELINE_SIZE[0])
            draw_h = int(src_h * display_size[1] / PIPELINE_SIZE[1])
            osd_img.draw_rectangle((draw_x, draw_y, draw_w, draw_h), color=(255, 0, 0), thickness=2)
            valid_count += 1

        draw_label(osd_img, 8, 8, "CV2 COMPAT TEST", (0, 255, 0))
        draw_label(osd_img, 8, 36, "CHW2HWC:{}ms CV:{}ms".format(convert_ms, cv_ms), (255, 255, 0))
        draw_label(osd_img, 8, 64, "RED:{} MEM:{}".format(valid_count, free_memory_text()), (255, 0, 0))
        pl.show_image()

        if not first_shape_printed:
            print("HWC shape:", frame_hwc.shape)
            print("CV image shape:", small_rgb.shape)
            first_shape_printed = True

        if frame_count % 30 == 0:
            print("30-frame average: chw2hwc={}ms cv={}ms red={} free_mem={}".format(
                total_convert_ms / 30,
                total_cv_ms / 30,
                valid_count,
                free_memory_text(),
            ))
            total_convert_ms = 0
            total_cv_ms = 0
            gc.collect()

        del frame_hwc, small_rgb, hsv, mask_1, mask_2, mask, contours
        time.sleep_ms(10)

except KeyboardInterrupt as e:
    print("user stopped:", e)
except BaseException as e:
    print("compatibility test FAILED:", e)
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
    print("compatibility test finished")
