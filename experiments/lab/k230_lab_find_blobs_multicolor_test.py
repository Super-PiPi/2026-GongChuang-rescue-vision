# ============================================================
# K230 CanMV/OpenMV LAB 多色色块独立测试
# 当前启用红、蓝、黄；黑色因与阴影/暗背景不可分，默认关闭并继续交给 YOLO。
# 直接使用 RGB565 + find_blobs，不使用 cv2、不加载 YOLO、不发送串口命令。
# ============================================================

import gc
import os
import time

from media.sensor import *
from media.display import *
from media.media import *


SENSOR_ID = 1
DETECT_WIDTH = 320
DETECT_HEIGHT = 240
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480

# CanMV IDE LAB 阈值编辑器结果：
# (L Min, L Max, A Min, A Max, B Min, B Max)
RED_LAB_THRESHOLD = (18, 54, 30, 60, -6, 51)
BLUE_LAB_THRESHOLD = (19, 50, 0, 59, -82, -33)
YELLOW_LAB_THRESHOLD = (38, 83, -29, -5, 34, 122)

# 该范围来自当前黑球截图，但会同时选中阴影、衣物、桌底等暗区域。
# 只用于后续受限实验，默认不要启用。
BLACK_LAB_THRESHOLD = (9, 23, -11, 7, -9, 23)
ENABLE_BLACK_LAB = False  # True/False 可现场切换，默认关闭黑色阈值。

COLOR_CONFIGS = [
    ("RED", RED_LAB_THRESHOLD, (255, 0, 0)),
    ("BLUE", BLUE_LAB_THRESHOLD, (0, 0, 255)),
    ("YELLOW", YELLOW_LAB_THRESHOLD, (255, 255, 0)),
]
if ENABLE_BLACK_LAB:
    COLOR_CONFIGS.append(("BLACK", BLACK_LAB_THRESHOLD, (255, 255, 255)))

# find_blobs 初筛参数，全部基于 320x240 图像。
PIXELS_THRESHOLD = 40
AREA_THRESHOLD = 50
MERGE_BLOBS = False
MERGE_MARGIN = 0

# 二次过滤参数。先保持与已验证的蓝色测试一致。
MIN_BOX_AREA = 50
MAX_BOX_AREA = 30000
MIN_BOX_WIDTH = 5
MIN_BOX_HEIGHT = 5
MIN_DENSITY = 0.25
MAX_ASPECT_RATIO = 4.0


def is_valid_blob(blob):
    width = blob.w()
    height = blob.h()
    box_area = width * height

    if box_area < MIN_BOX_AREA or box_area > MAX_BOX_AREA:
        return False
    if width < MIN_BOX_WIDTH or height < MIN_BOX_HEIGHT:
        return False
    if blob.density() < MIN_DENSITY:
        return False

    aspect_ratio = width / height
    if aspect_ratio < 1.0:
        aspect_ratio = 1.0 / aspect_ratio
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False
    return True


def decode_blob_color(blob_code):
    matched = []
    for index in range(len(COLOR_CONFIGS)):
        if blob_code & (1 << index):
            matched.append(index)

    if not matched:
        return "UNKNOWN", (255, 255, 255)
    if len(matched) == 1:
        config = COLOR_CONFIGS[matched[0]]
        return config[0], config[2]

    # merge=False 时通常只有一个 bit；若阈值发生重叠则明确标记 MIXED。
    return "MIXED", (255, 0, 255)


sensor = None

try:
    os.exitpoint(os.EXITPOINT_ENABLE)
    print("===== K230 LAB find_blobs multi-color test =====")
    print("detect size:{}x{} black_enabled:{}".format(
        DETECT_WIDTH, DETECT_HEIGHT, ENABLE_BLACK_LAB
    ))
    for index in range(len(COLOR_CONFIGS)):
        print("code={} color={} threshold={}".format(
            1 << index, COLOR_CONFIGS[index][0], COLOR_CONFIGS[index][1]
        ))

    sensor = Sensor(id=SENSOR_ID)
    sensor.reset()
    sensor.set_framesize(width=DETECT_WIDTH, height=DETECT_HEIGHT)
    sensor.set_pixformat(Sensor.RGB565)

    Display.init(
        Display.ST7701,
        width=DISPLAY_WIDTH,
        height=DISPLAY_HEIGHT,
        to_ide=True,
    )
    MediaManager.init()
    sensor.run()

    thresholds = []
    for config in COLOR_CONFIGS:
        thresholds.append(config[1])

    fps_clock = time.clock()
    frame_count = 0
    total_find_ms = 0

    while True:
        os.exitpoint()
        fps_clock.tick()
        img = sensor.snapshot()

        start_ms = time.ticks_ms()
        blobs = img.find_blobs(
            thresholds,
            pixels_threshold=PIXELS_THRESHOLD,
            area_threshold=AREA_THRESHOLD,
            merge=MERGE_BLOBS,
            margin=MERGE_MARGIN,
        )
        find_ms = time.ticks_diff(time.ticks_ms(), start_ms)
        total_find_ms += find_ms
        frame_count += 1

        valid_results = []
        color_counts = {}
        for config in COLOR_CONFIGS:
            color_counts[config[0]] = 0
        color_counts["MIXED"] = 0

        for blob in blobs:
            if not is_valid_blob(blob):
                continue

            color_name, draw_color = decode_blob_color(blob.code())
            valid_results.append((color_name, blob))
            if color_name not in color_counts:
                color_counts[color_name] = 0
            color_counts[color_name] += 1

            img.draw_rectangle(blob.rect(), color=draw_color, thickness=2)
            img.draw_cross(blob.cx(), blob.cy(), color=(255, 255, 255), size=8)
            label_y = blob.y() - 16
            if label_y < 0:
                label_y = blob.y()
            try:
                img.draw_string(
                    blob.x(),
                    label_y,
                    "{} P:{}".format(color_name, blob.pixels()),
                    color=draw_color,
                    scale=1,
                )
            except Exception:
                pass

        count_text = "R:{} B:{} Y:{}".format(
            color_counts.get("RED", 0),
            color_counts.get("BLUE", 0),
            color_counts.get("YELLOW", 0),
        )
        if ENABLE_BLACK_LAB:
            count_text += " K:{}".format(color_counts.get("BLACK", 0))
        img.draw_string(
            2,
            2,
            "{} {}ms".format(count_text, find_ms),
            color=(255, 255, 255),
            scale=1,
        )

        Display.show_image(
            img,
            x=(DISPLAY_WIDTH - DETECT_WIDTH) // 2,
            y=(DISPLAY_HEIGHT - DETECT_HEIGHT) // 2,
        )

        if frame_count % 30 == 0:
            print("frame={} fps={:.2f} avg_find={}ms raw={} valid={} counts={}".format(
                frame_count,
                fps_clock.fps(),
                total_find_ms / 30,
                len(blobs),
                len(valid_results),
                color_counts,
            ))
            for color_name, blob in valid_results:
                main_cx = int(blob.cx() * 800 / DETECT_WIDTH)
                main_cy = int(blob.cy() * 480 / DETECT_HEIGHT)
                main_w = int(blob.w() * 800 / DETECT_WIDTH)
                main_h = int(blob.h() * 480 / DETECT_HEIGHT)
                print("  {} main_center=({}, {}) main_wh=({}, {}) pixels={} density={:.2f} code={}".format(
                    color_name,
                    main_cx,
                    main_cy,
                    main_w,
                    main_h,
                    blob.pixels(),
                    blob.density(),
                    blob.code(),
                ))
            total_find_ms = 0
            gc.collect()

        time.sleep_ms(5)

except KeyboardInterrupt as e:
    print("user stopped:", e)
except BaseException as e:
    if str(e) == "IDE interrupt":
        print("user stopped by IDE")
    else:
        print("LAB multi-color test FAILED:", e)
finally:
    try:
        if sensor is not None:
            sensor.stop()
    except Exception:
        pass
    try:
        Display.deinit()
    except Exception:
        pass
    try:
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    gc.collect()
    print("LAB multi-color test finished")

