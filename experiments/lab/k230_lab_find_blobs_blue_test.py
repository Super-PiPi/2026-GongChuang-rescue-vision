# ============================================================
# K230 CanMV/OpenMV LAB 蓝色色块独立测试
# 直接使用 RGB565 + find_blobs，与 CanMV IDE LAB 阈值编辑器保持一致。
# 不加载 YOLO、不使用 cv2、不发送串口命令。
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

# CanMV IDE LAB 阈值编辑器得到的蓝色阈值：
# (L Min, L Max, A Min, A Max, B Min, B Max)
BLUE_LAB_THRESHOLD = (19, 50, 0, 59, -82, -33)

# find_blobs 初筛参数，全部基于 320x240 图像。
PIXELS_THRESHOLD = 40
AREA_THRESHOLD = 50
MERGE_BLOBS = False
MERGE_MARGIN = 0

# 二次过滤，先保持宽松，观察 LAB 阈值本身的效果。
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

    density = blob.density()
    if density < MIN_DENSITY:
        return False

    aspect_ratio = width / height
    if aspect_ratio < 1.0:
        aspect_ratio = 1.0 / aspect_ratio
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False

    return True


sensor = None

try:
    os.exitpoint(os.EXITPOINT_ENABLE)
    print("===== K230 LAB find_blobs BLUE test =====")
    print("threshold:", BLUE_LAB_THRESHOLD)
    print("detect size:{}x{}".format(DETECT_WIDTH, DETECT_HEIGHT))

    # 直接输出 RGB565，保证 LAB 阈值与 CanMV IDE 编辑器/find_blobs 语义一致。
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

    fps_clock = time.clock()
    frame_count = 0
    total_find_ms = 0

    while True:
        os.exitpoint()
        fps_clock.tick()
        img = sensor.snapshot()

        start_ms = time.ticks_ms()
        blobs = img.find_blobs(
            [BLUE_LAB_THRESHOLD],
            pixels_threshold=PIXELS_THRESHOLD,
            area_threshold=AREA_THRESHOLD,
            merge=MERGE_BLOBS,
            margin=MERGE_MARGIN,
        )
        find_ms = time.ticks_diff(time.ticks_ms(), start_ms)
        total_find_ms += find_ms
        frame_count += 1

        valid_blobs = []
        for blob in blobs:
            if not is_valid_blob(blob):
                continue

            valid_blobs.append(blob)
            img.draw_rectangle(blob.rect(), color=(255, 0, 0), thickness=2)
            img.draw_cross(blob.cx(), blob.cy(), color=(255, 255, 255), size=8)
            label_y = blob.y() - 16
            if label_y < 0:
                label_y = blob.y()
            img.draw_string(
                blob.x(),
                label_y,
                "BLUE P:{} D:{:.2f}".format(blob.pixels(), blob.density()),
                color=(255, 255, 255),
                scale=1,
            )

        img.draw_string(
            2,
            2,
            "LAB BLUE N:{} {}ms".format(len(valid_blobs), find_ms),
            color=(255, 255, 0),
            scale=1,
        )

        # 将 320x240 调试画面居中显示在 800x480 LCD，同时发送到 IDE。
        Display.show_image(
            img,
            x=(DISPLAY_WIDTH - DETECT_WIDTH) // 2,
            y=(DISPLAY_HEIGHT - DETECT_HEIGHT) // 2,
        )

        if frame_count % 30 == 0:
            print("frame={} fps={:.2f} avg_find={}ms raw={} valid={}".format(
                frame_count,
                fps_clock.fps(),
                total_find_ms / 30,
                len(blobs),
                len(valid_blobs),
            ))
            for blob in valid_blobs:
                # 同时给出映射回主程序 800x480 坐标后的中心点。
                main_cx = int(blob.cx() * 800 / DETECT_WIDTH)
                main_cy = int(blob.cy() * 480 / DETECT_HEIGHT)
                main_w = int(blob.w() * 800 / DETECT_WIDTH)
                main_h = int(blob.h() * 480 / DETECT_HEIGHT)
                print("  BLUE cv=({}, {}, {}, {}) main_center=({}, {}) main_wh=({}, {}) pixels={} density={:.2f}".format(
                    blob.x(),
                    blob.y(),
                    blob.w(),
                    blob.h(),
                    main_cx,
                    main_cy,
                    main_w,
                    main_h,
                    blob.pixels(),
                    blob.density(),
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
        print("LAB find_blobs test FAILED:", e)
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
    print("LAB find_blobs test finished")
