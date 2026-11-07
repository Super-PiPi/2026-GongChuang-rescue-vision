# ============================================================
# K230 OpenCV HSV 环境标定脚本
# 将待标定物体尽量填满屏幕中央取样框，自动采样并输出建议 HSV 阈值。
# 输出包含原始范围、5%~95%抗离群范围和可直接复制的 HSV_RANGES 配置。
# ============================================================

import gc
import math
import os
import time

import cv2
from ulab import numpy as np
from media.sensor import Sensor
from media.media import MediaManager
from libs.PipeLine import PipeLine
from libs.Utils import chw2hwc


# ===================== 只需要优先调整这里 =====================
# 每次标定一种颜色，依次改成 RED、BLUE、YELLOW、BLACK 后重新运行。
SAMPLE_NAME = "RED"

SENSOR_ID = 1
PIPELINE_SIZE = [800, 480]
CV_SIZE = [320, 240]
DISPLAY_MODE = "lcd"

# 中央取样框尺寸，单位为 320x240 CV 图像像素。
# 60x60 对应原始 800x480 画面中的约 150x120 区域。
ROI_WIDTH = 60
ROI_HEIGHT = 60

# 启动后先等待摄像头曝光/白平衡稳定，并留时间把目标放进取样框。
START_DELAY_MS = 5000
SAMPLE_FRAME_COUNT = 40

# 建议阈值在统计范围外再放宽的余量。
H_MARGIN = 3
S_MARGIN = 15
V_MARGIN = 15
BLACK_V_MARGIN = 10

# 计算彩色物体的 H 色相时，忽略低饱和、过暗像素，避免灰色和黑色干扰 H。
HUE_VALID_MIN_S = 40
HUE_VALID_MIN_V = 40


def draw_label(img, x, y, text, color=(0, 255, 0), char_size=22):
    img.draw_string_advanced(x + 1, y + 1, char_size, text, (0, 0, 0))
    img.draw_string_advanced(x, y, char_size, text, color)


def clamp(value, low, high):
    value = int(value)
    if value < low:
        return low
    if value > high:
        return high
    return value


def hist_value(hist, index):
    """兼容直方图为 (N,) 或 (N,1) 两种 ndarray 形状。"""
    value = hist[index]
    try:
        return float(value[0])
    except Exception:
        return float(value)


def histogram_total(hist, bins):
    total = 0.0
    for i in range(bins):
        total += hist_value(hist, i)
    return total


def histogram_percentile(hist, bins, percentile):
    total = histogram_total(hist, bins)
    if total <= 0:
        return 0

    target = total * percentile
    accumulated = 0.0
    for i in range(bins):
        accumulated += hist_value(hist, i)
        if accumulated >= target:
            return i
    return bins - 1


def histogram_peak(hist, bins):
    best_index = 0
    best_value = -1.0
    for i in range(bins):
        value = hist_value(hist, i)
        if value > best_value:
            best_value = value
            best_index = i
    return best_index


def shortest_circular_hue_interval(hist, coverage=0.90):
    """寻找覆盖指定比例有效像素的最短圆环色相区间，正确处理红色跨 0/179。"""
    counts = []
    total = 0.0
    for i in range(180):
        value = hist_value(hist, i)
        counts.append(value)
        total += value

    if total <= 0:
        return 0, 179, 180

    target = total * coverage
    best_start = 0
    best_length = 180

    for start in range(180):
        accumulated = 0.0
        for offset in range(180):
            accumulated += counts[(start + offset) % 180]
            if accumulated >= target:
                length = offset + 1
                if length < best_length:
                    best_start = start
                    best_length = length
                break

    best_end = (best_start + best_length - 1) % 180
    return best_start, best_end, best_length


def expand_circular_hue_interval(start, length, margin):
    expanded_length = length + margin * 2
    if expanded_length >= 180:
        return [(0, 179)]

    expanded_start = (start - margin) % 180
    expanded_end_unwrapped = expanded_start + expanded_length - 1
    if expanded_end_unwrapped < 180:
        return [(expanded_start, expanded_end_unwrapped)]

    # 为便于复制到检测程序，将跨 0 的范围按低 H、再高 H 的顺序输出。
    return [(0, expanded_end_unwrapped - 180), (expanded_start, 179)]


def add_histogram(sum_hist, new_hist):
    if sum_hist is None:
        return new_hist
    return cv2.add(sum_hist, new_hist)


def format_range_line(color_name, h_low, h_high, s_low, v_low, v_high=255):
    return (
        "(np.array([{}, {}, {}], dtype=np.uint8), "
        "np.array([{}, 255, {}], dtype=np.uint8)),"
    ).format(h_low, s_low, v_low, h_high, v_high)


def print_final_result(result):
    print("")
    print("============================================================")
    print("HSV CALIBRATION RESULT: {}".format(SAMPLE_NAME))
    print("sample frames:{} ROI:{}x{} valid hue pixels:{}".format(
        SAMPLE_FRAME_COUNT,
        ROI_WIDTH,
        ROI_HEIGHT,
        int(result["valid_hue_pixels"]),
    ))
    print("raw min: H={} S={} V={}".format(
        result["raw_min"][0], result["raw_min"][1], result["raw_min"][2]
    ))
    print("raw max: H={} S={} V={}".format(
        result["raw_max"][0], result["raw_max"][1], result["raw_max"][2]
    ))
    print("mean   : H={:.1f} S={:.1f} V={:.1f}".format(
        result["mean"][0], result["mean"][1], result["mean"][2]
    ))
    print("P05    : H={} S={} V={}".format(
        result["p05"][0], result["p05"][1], result["p05"][2]
    ))
    print("P50    : H={} S={} V={}".format(
        result["p50"][0], result["p50"][1], result["p50"][2]
    ))
    print("P95    : H={} S={} V={}".format(
        result["p95"][0], result["p95"][1], result["p95"][2]
    ))
    print("H peak : {}".format(result["h_peak"]))
    print("H circular 90%: start={} end={} length={}".format(
        result["h_circular_start"],
        result["h_circular_end"],
        result["h_circular_length"],
    ))

    print("")
    print("Suggested HSV_RANGES entry:")
    print('    "{}": ('.format(SAMPLE_NAME))
    for line in result["config_lines"]:
        print("        " + line)
    print("    ),")
    print("============================================================")
    print("Keep the object in the ROI and save the complete output above.")


def summarize_calibration(hist_h, hist_s, hist_v, raw_min, raw_max, mean_sum, sample_count):
    p05 = (
        histogram_percentile(hist_h, 180, 0.05),
        histogram_percentile(hist_s, 256, 0.05),
        histogram_percentile(hist_v, 256, 0.05),
    )
    p50 = (
        histogram_percentile(hist_h, 180, 0.50),
        histogram_percentile(hist_s, 256, 0.50),
        histogram_percentile(hist_v, 256, 0.50),
    )
    p95 = (
        histogram_percentile(hist_h, 180, 0.95),
        histogram_percentile(hist_s, 256, 0.95),
        histogram_percentile(hist_v, 256, 0.95),
    )

    h_start, h_end, h_length = shortest_circular_hue_interval(hist_h, 0.90)
    h_ranges = expand_circular_hue_interval(h_start, h_length, H_MARGIN)
    mean_values = (
        mean_sum[0] / sample_count,
        mean_sum[1] / sample_count,
        mean_sum[2] / sample_count,
    )

    config_lines = []
    if SAMPLE_NAME == "BLACK":
        # 黑色主要看亮度 V，H 没有实际意义，S 也不应限制过紧。
        black_v_high = clamp(p95[2] + BLACK_V_MARGIN, 0, 255)
        config_lines.append(format_range_line(SAMPLE_NAME, 0, 179, 0, 0, black_v_high))
    else:
        s_low = clamp(p05[1] - S_MARGIN, 0, 255)
        v_low = clamp(p05[2] - V_MARGIN, 0, 255)
        for h_low, h_high in h_ranges:
            config_lines.append(format_range_line(
                SAMPLE_NAME, h_low, h_high, s_low, v_low, 255
            ))

    return {
        "raw_min": tuple(raw_min),
        "raw_max": tuple(raw_max),
        "mean": mean_values,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "h_peak": histogram_peak(hist_h, 180),
        "h_circular_start": h_start,
        "h_circular_end": h_end,
        "h_circular_length": h_length,
        "valid_hue_pixels": histogram_total(hist_h, 180),
        "config_lines": config_lines,
        "h_ranges": h_ranges,
    }


pl = None
sensor = None

try:
    os.exitpoint(os.EXITPOINT_ENABLE)
    print("===== K230 HSV calibration =====")
    print("sample name:", SAMPLE_NAME)
    print("Place the object inside the center box before the countdown ends.")

    required_apis = ("resize", "cvtColor", "calcHist", "inRange", "split", "mean", "minMaxLoc")
    missing_apis = []
    for api_name in required_apis:
        if not hasattr(cv2, api_name):
            missing_apis.append(api_name)
    if missing_apis:
        raise RuntimeError("current firmware lacks cv2 APIs: {}".format(missing_apis))

    pl = PipeLine(
        rgb888p_size=PIPELINE_SIZE,
        display_mode=DISPLAY_MODE,
        display_size=None,
    )
    sensor = Sensor(id=SENSOR_ID)
    pl.create(sensor)
    display_size = pl.get_display_size()

    roi_x = (CV_SIZE[0] - ROI_WIDTH) // 2
    roi_y = (CV_SIZE[1] - ROI_HEIGHT) // 2
    src_roi = (
        int(roi_x * PIPELINE_SIZE[0] / CV_SIZE[0]),
        int(roi_y * PIPELINE_SIZE[1] / CV_SIZE[1]),
        int(ROI_WIDTH * PIPELINE_SIZE[0] / CV_SIZE[0]),
        int(ROI_HEIGHT * PIPELINE_SIZE[1] / CV_SIZE[1]),
    )
    display_roi = (
        int(src_roi[0] * display_size[0] / PIPELINE_SIZE[0]),
        int(src_roi[1] * display_size[1] / PIPELINE_SIZE[1]),
        int(src_roi[2] * display_size[0] / PIPELINE_SIZE[0]),
        int(src_roi[3] * display_size[1] / PIPELINE_SIZE[1]),
    )

    start_ms = time.ticks_ms()
    sample_count = 0
    hist_h_sum = None
    hist_s_sum = None
    hist_v_sum = None
    raw_min = [179, 255, 255]
    raw_max = [0, 0, 0]
    mean_sum = [0.0, 0.0, 0.0]
    final_result = None
    shape_printed = False

    hue_mask_low = np.array([0, HUE_VALID_MIN_S, HUE_VALID_MIN_V], dtype=np.uint8)
    hue_mask_high = np.array([179, 255, 255], dtype=np.uint8)

    while True:
        os.exitpoint()
        frame_chw = pl.get_frame()
        osd_img = pl.osd_img
        osd_img.clear()

        osd_img.draw_rectangle(display_roi, color=(0, 255, 0), thickness=3)
        center_x = display_roi[0] + display_roi[2] // 2
        center_y = display_roi[1] + display_roi[3] // 2
        osd_img.draw_cross(center_x, center_y, color=(0, 255, 0), size=12)
        draw_label(osd_img, 8, 8, "HSV CAL:{}".format(SAMPLE_NAME), (0, 255, 0))

        elapsed_ms = time.ticks_diff(time.ticks_ms(), start_ms)

        if elapsed_ms < START_DELAY_MS:
            remain_ms = START_DELAY_MS - elapsed_ms
            draw_label(osd_img, 8, 36, "PLACE OBJECT: {:.1f}s".format(remain_ms / 1000), (255, 255, 0))
            draw_label(osd_img, 8, 64, "FILL THE CENTER BOX", (255, 255, 255))

        elif sample_count < SAMPLE_FRAME_COUNT:
            process_start_ms = time.ticks_ms()
            frame_hwc = chw2hwc(frame_chw)
            small_rgb = cv2.resize(
                frame_hwc,
                (CV_SIZE[0], CV_SIZE[1]),
                interpolation=cv2.INTER_AREA,
            )
            hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
            hsv_roi = hsv[roi_y:roi_y + ROI_HEIGHT, roi_x:roi_x + ROI_WIDTH, :]

            hue_valid_mask = cv2.inRange(hsv_roi, hue_mask_low, hue_mask_high)
            hist_h = cv2.calcHist([hsv_roi], [0], hue_valid_mask, [180], [0, 180])
            hist_s = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv_roi], [2], None, [256], [0, 256])
            hist_h_sum = add_histogram(hist_h_sum, hist_h)
            hist_s_sum = add_histogram(hist_s_sum, hist_s)
            hist_v_sum = add_histogram(hist_v_sum, hist_v)

            channels = cv2.split(hsv_roi)
            for channel_index in range(3):
                min_value, max_value, _, _ = cv2.minMaxLoc(channels[channel_index])
                if min_value < raw_min[channel_index]:
                    raw_min[channel_index] = int(min_value)
                if max_value > raw_max[channel_index]:
                    raw_max[channel_index] = int(max_value)

            mean_values = cv2.mean(hsv_roi)
            mean_sum[0] += mean_values[0]
            mean_sum[1] += mean_values[1]
            mean_sum[2] += mean_values[2]
            sample_count += 1
            process_ms = time.ticks_diff(time.ticks_ms(), process_start_ms)

            draw_label(osd_img, 8, 36, "SAMPLING:{}/{}".format(
                sample_count, SAMPLE_FRAME_COUNT
            ), (255, 255, 0))
            draw_label(osd_img, 8, 64, "MEAN H:{:.0f} S:{:.0f} V:{:.0f}".format(
                mean_values[0], mean_values[1], mean_values[2]
            ), (255, 255, 255))
            draw_label(osd_img, 8, 92, "PROCESS:{}ms".format(process_ms), (0, 255, 255))

            if not shape_printed:
                print("CHW shape:", frame_chw.shape)
                print("HWC shape:", frame_hwc.shape)
                print("CV shape:", small_rgb.shape)
                print("HSV ROI shape:", hsv_roi.shape)
                shape_printed = True

            del frame_hwc, small_rgb, hsv, hsv_roi, hue_valid_mask
            del hist_h, hist_s, hist_v, channels

            if sample_count >= SAMPLE_FRAME_COUNT:
                final_result = summarize_calibration(
                    hist_h_sum,
                    hist_s_sum,
                    hist_v_sum,
                    raw_min,
                    raw_max,
                    mean_sum,
                    sample_count,
                )
                print_final_result(final_result)
                gc.collect()

        else:
            draw_label(osd_img, 8, 36, "CALIBRATION COMPLETE", (0, 255, 0))
            draw_label(osd_img, 8, 64, "CHECK SERIAL OUTPUT", (255, 255, 255))
            if final_result is not None:
                draw_label(osd_img, 8, 92, "P05 S:{} V:{}  HPEAK:{}".format(
                    final_result["p05"][1],
                    final_result["p05"][2],
                    final_result["h_peak"],
                ), (0, 255, 255))

        pl.show_image()
        time.sleep_ms(10)

except KeyboardInterrupt as e:
    print("user stopped:", e)
except BaseException as e:
    if str(e) == "IDE interrupt":
        print("user stopped by IDE")
    else:
        print("HSV calibration FAILED:", e)
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
    print("HSV calibration finished")

