# ============================================================
# K230 OpenCV 交互式 HSV 快速标定器
# 运行一次即可移动/缩放 ROI，采集红蓝黄黑多组样本并输出最终 HSV_RANGES。
# 结果同时打印到串口并保存到 /sdcard/hsv_calibration.txt。
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


# ===================== 基本配置 =====================
SENSOR_ID = 1
PIPELINE_SIZE = [800, 480]
CV_SIZE = [320, 240]
DISPLAY_MODE = "lcd"
SAVE_PATH = "/sdcard/hsv_calibration.txt"

# ROI 坐标和尺寸均基于 320x240 CV 图像。
ROI_INITIAL_SIZE = 12
ROI_MIN_SIZE = 4
ROI_MAX_SIZE = 60
ROI_MOVE_STEP = 8
ROI_SIZE_STEP = 4

# 每按一次 C，累积采集若干帧。10 帧通常不到 1 秒。
CAPTURE_FRAME_COUNT = 10

# 阈值输出余量。最终阈值基于累计样本的稳健统计范围再适度放宽。
H_MARGIN = 3
S_MARGIN = 15
V_MARGIN = 15
BLACK_V_MARGIN = 10

# 彩色目标统计 H 时忽略低饱和、过暗像素，避免灰色高光和阴影污染色相。
HUE_VALID_MIN_S = 40
HUE_VALID_MIN_V = 40

COLOR_NAMES = ("RED", "BLUE", "YELLOW", "BLACK")
OSD_COLORS = {
    "RED": (255, 0, 0),
    "BLUE": (0, 0, 255),
    "YELLOW": (255, 255, 0),
    "BLACK": (255, 255, 255),
}


def draw_label(img, x, y, text, color=(0, 255, 0), char_size=20):
    img.draw_string_advanced(x + 1, y + 1, char_size, text, (0, 0, 0))
    img.draw_string_advanced(x, y, char_size, text, color)


def clamp(value, low, high):
    value = int(value)
    if value < low:
        return low
    if value > high:
        return high
    return value


def new_color_stats():
    return {
        "hist_h": None,
        "hist_s": None,
        "hist_v": None,
        "raw_min": [179, 255, 255],
        "raw_max": [0, 0, 0],
        "mean_sum": [0.0, 0.0, 0.0],
        "frames": 0,
        "captures": 0,
    }


def reset_color_stats(all_stats, color_name):
    all_stats[color_name] = new_color_stats()
    gc.collect()


def hist_value(hist, index):
    # 兼容 calcHist 返回 (N,) 或 (N,1) 两种形状。
    value = hist[index]
    try:
        return float(value[0])
    except Exception:
        return float(value)


def histogram_total(hist, bins):
    if hist is None:
        return 0.0
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
    if hist is None:
        return 0
    best_index = 0
    best_value = -1.0
    for i in range(bins):
        value = hist_value(hist, i)
        if value > best_value:
            best_value = value
            best_index = i
    return best_index


def shortest_circular_hue_interval(hist, coverage=0.90):
    # H 是 0~179 的圆环，红色可能同时分布在 0 和 179 附近。
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

    return best_start, (best_start + best_length - 1) % 180, best_length


def expand_circular_hue_interval(start, length, margin):
    expanded_length = length + margin * 2
    if expanded_length >= 180:
        return [(0, 179)]

    expanded_start = (start - margin) % 180
    expanded_end_unwrapped = expanded_start + expanded_length - 1
    if expanded_end_unwrapped < 180:
        return [(expanded_start, expanded_end_unwrapped)]

    # 跨越 0/179 时按低 H 段、高 H 段的顺序输出，便于直接复制。
    return [(0, expanded_end_unwrapped - 180), (expanded_start, 179)]


def add_histogram(sum_hist, new_hist):
    if sum_hist is None:
        return new_hist
    return cv2.add(sum_hist, new_hist)


def get_roi_bounds(roi_cx, roi_cy, roi_size):
    half = roi_size // 2
    x = clamp(roi_cx - half, 0, CV_SIZE[0] - roi_size)
    y = clamp(roi_cy - half, 0, CV_SIZE[1] - roi_size)
    return x, y, roi_size, roi_size


def cv_roi_to_display(cv_roi, display_size):
    x, y, w, h = cv_roi
    src_x = int(x * PIPELINE_SIZE[0] / CV_SIZE[0])
    src_y = int(y * PIPELINE_SIZE[1] / CV_SIZE[1])
    src_w = int(w * PIPELINE_SIZE[0] / CV_SIZE[0])
    src_h = int(h * PIPELINE_SIZE[1] / CV_SIZE[1])
    return (
        int(src_x * display_size[0] / PIPELINE_SIZE[0]),
        int(src_y * display_size[1] / PIPELINE_SIZE[1]),
        int(src_w * display_size[0] / PIPELINE_SIZE[0]),
        int(src_h * display_size[1] / PIPELINE_SIZE[1]),
    )


def calculate_roi_frame_data(hsv_roi, hue_mask_low, hue_mask_high):
    hue_valid_mask = cv2.inRange(hsv_roi, hue_mask_low, hue_mask_high)
    hist_h = cv2.calcHist([hsv_roi], [0], hue_valid_mask, [180], [0, 180])
    hist_s = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv_roi], [2], None, [256], [0, 256])
    means = cv2.mean(hsv_roi)
    return hist_h, hist_s, hist_v, means, hue_valid_mask


def add_frame_to_color_stats(stats, hsv_roi, hist_h, hist_s, hist_v, means):
    stats["hist_h"] = add_histogram(stats["hist_h"], hist_h)
    stats["hist_s"] = add_histogram(stats["hist_s"], hist_s)
    stats["hist_v"] = add_histogram(stats["hist_v"], hist_v)

    channels = cv2.split(hsv_roi)
    for index in range(3):
        min_value, max_value, _, _ = cv2.minMaxLoc(channels[index])
        if min_value < stats["raw_min"][index]:
            stats["raw_min"][index] = int(min_value)
        if max_value > stats["raw_max"][index]:
            stats["raw_max"][index] = int(max_value)

    stats["mean_sum"][0] += means[0]
    stats["mean_sum"][1] += means[1]
    stats["mean_sum"][2] += means[2]
    stats["frames"] += 1
    del channels


def make_config_line(h_low, h_high, s_low, v_low, v_high=255):
    return (
        "(np.array([{}, {}, {}], dtype=np.uint8), "
        "np.array([{}, 255, {}], dtype=np.uint8)),"
    ).format(h_low, s_low, v_low, h_high, v_high)


def summarize_color(color_name, stats):
    if stats["frames"] <= 0:
        return None

    p05 = (
        histogram_percentile(stats["hist_h"], 180, 0.05),
        histogram_percentile(stats["hist_s"], 256, 0.05),
        histogram_percentile(stats["hist_v"], 256, 0.05),
    )
    p50 = (
        histogram_percentile(stats["hist_h"], 180, 0.50),
        histogram_percentile(stats["hist_s"], 256, 0.50),
        histogram_percentile(stats["hist_v"], 256, 0.50),
    )
    p95 = (
        histogram_percentile(stats["hist_h"], 180, 0.95),
        histogram_percentile(stats["hist_s"], 256, 0.95),
        histogram_percentile(stats["hist_v"], 256, 0.95),
    )
    h_start, h_end, h_length = shortest_circular_hue_interval(stats["hist_h"], 0.90)
    h_ranges = expand_circular_hue_interval(h_start, h_length, H_MARGIN)

    config_lines = []
    if color_name == "BLACK":
        black_v_high = clamp(p95[2] + BLACK_V_MARGIN, 0, 255)
        config_lines.append(make_config_line(0, 179, 0, 0, black_v_high))
    else:
        s_low = clamp(p05[1] - S_MARGIN, 0, 255)
        v_low = clamp(p05[2] - V_MARGIN, 0, 255)
        for h_low, h_high in h_ranges:
            config_lines.append(make_config_line(h_low, h_high, s_low, v_low, 255))

    return {
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "h_peak": histogram_peak(stats["hist_h"], 180),
        "h_start": h_start,
        "h_end": h_end,
        "h_length": h_length,
        "config_lines": config_lines,
    }


def build_output_text(all_stats):
    lines = []
    lines.append("============================================================")
    lines.append("K230 HSV QUICK CALIBRATION RESULT")
    lines.append("HSV_RANGES = {")

    completed_colors = 0
    for color_name in COLOR_NAMES:
        stats = all_stats[color_name]
        result = summarize_color(color_name, stats)
        if result is None:
            lines.append('    # "{}": no samples'.format(color_name))
            continue

        completed_colors += 1
        lines.append("    # {} captures={} frames={}".format(
            color_name, stats["captures"], stats["frames"]
        ))
        lines.append("    # raw min={} raw max={}".format(
            stats["raw_min"], stats["raw_max"]
        ))
        lines.append("    # P05={} P50={} P95={}".format(
            result["p05"], result["p50"], result["p95"]
        ))
        lines.append("    # H peak={} circular90=({}, {}, length={})".format(
            result["h_peak"], result["h_start"], result["h_end"], result["h_length"]
        ))
        lines.append('    "{}": ('.format(color_name))
        for config_line in result["config_lines"]:
            lines.append("        " + config_line)
        lines.append("    ),")

    lines.append("}")
    lines.append("completed colors: {}/{}".format(completed_colors, len(COLOR_NAMES)))
    lines.append("============================================================")
    return "\n".join(lines)


def print_and_save_results(all_stats):
    output_text = build_output_text(all_stats)
    print("")
    print(output_text)
    try:
        with open(SAVE_PATH, "w") as output_file:
            output_file.write(output_text)
            output_file.write("\n")
        print("saved to:", SAVE_PATH)
    except Exception as e:
        print("save failed:", e)


def print_help():
    print("")
    print("Interactive keys (click the serial terminal before pressing):")
    print("  r/b/y/k : select RED/BLUE/YELLOW/BLACK")
    print("  w/a/s/d : move ROI up/left/down/right")
    print("  + or =  : enlarge ROI")
    print("  -       : shrink ROI")
    print("  c       : capture {} frames for current color".format(CAPTURE_FRAME_COUNT))
    print("  x       : clear current color samples")
    print("  p       : print and save all HSV ranges")
    print("  h       : print this help")
    print("  q       : quit")
    print("")


pl = None
sensor = None

try:
    os.exitpoint(os.EXITPOINT_ENABLE)
    print("===== K230 interactive HSV quick calibration =====")

    required_apis = (
        "resize", "cvtColor", "calcHist", "inRange", "split",
        "mean", "minMaxLoc", "add", "waitKey",
    )
    missing_apis = []
    for api_name in required_apis:
        if not hasattr(cv2, api_name):
            missing_apis.append(api_name)
    if missing_apis:
        raise RuntimeError("current firmware lacks cv2 APIs: {}".format(missing_apis))

    print_help()

    pl = PipeLine(
        rgb888p_size=PIPELINE_SIZE,
        display_mode=DISPLAY_MODE,
        display_size=None,
    )
    sensor = Sensor(id=SENSOR_ID)
    pl.create(sensor)
    display_size = pl.get_display_size()

    all_stats = {}
    for name in COLOR_NAMES:
        all_stats[name] = new_color_stats()

    current_color = "RED"
    roi_cx = CV_SIZE[0] // 2
    roi_cy = CV_SIZE[1] // 2
    roi_size = ROI_INITIAL_SIZE
    capture_color = None
    capture_remaining = 0
    last_key_text = "NONE"
    should_quit = False
    frame_count = 0

    hue_mask_low = np.array([0, HUE_VALID_MIN_S, HUE_VALID_MIN_V], dtype=np.uint8)
    hue_mask_high = np.array([179, 255, 255], dtype=np.uint8)

    while not should_quit:
        os.exitpoint()
        frame_chw = pl.get_frame()
        osd_img = pl.osd_img
        osd_img.clear()
        frame_count += 1

        process_start_ms = time.ticks_ms()
        frame_hwc = chw2hwc(frame_chw)
        small_rgb = cv2.resize(
            frame_hwc,
            (CV_SIZE[0], CV_SIZE[1]),
            interpolation=cv2.INTER_AREA,
        )
        hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)

        cv_roi = get_roi_bounds(roi_cx, roi_cy, roi_size)
        roi_x, roi_y, roi_w, roi_h = cv_roi
        hsv_roi = hsv[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, :]
        hist_h, hist_s, hist_v, means, hue_valid_mask = calculate_roi_frame_data(
            hsv_roi, hue_mask_low, hue_mask_high
        )

        live_h = histogram_percentile(hist_h, 180, 0.50)
        live_s = histogram_percentile(hist_s, 256, 0.50)
        live_v = histogram_percentile(hist_v, 256, 0.50)

        if capture_remaining > 0 and capture_color is not None:
            add_frame_to_color_stats(
                all_stats[capture_color], hsv_roi, hist_h, hist_s, hist_v, means
            )
            capture_remaining -= 1
            if capture_remaining == 0:
                all_stats[capture_color]["captures"] += 1
                print(">>> capture complete: {} captures={} frames={} live_hsv=({}, {}, {})".format(
                    capture_color,
                    all_stats[capture_color]["captures"],
                    all_stats[capture_color]["frames"],
                    live_h,
                    live_s,
                    live_v,
                ))
                capture_color = None
                gc.collect()

        display_roi = cv_roi_to_display(cv_roi, display_size)
        roi_color = OSD_COLORS[current_color]
        osd_img.draw_rectangle(display_roi, color=roi_color, thickness=3)
        osd_img.draw_cross(
            display_roi[0] + display_roi[2] // 2,
            display_roi[1] + display_roi[3] // 2,
            color=roi_color,
            size=10,
        )

        current_stats = all_stats[current_color]
        process_ms = time.ticks_diff(time.ticks_ms(), process_start_ms)
        draw_label(osd_img, 8, 8, "COLOR:{} ROI:{}".format(current_color, roi_size), roi_color)
        draw_label(osd_img, 8, 32, "LIVE H:{} S:{} V:{}".format(live_h, live_s, live_v), (0, 255, 255))
        draw_label(osd_img, 8, 56, "CAP:{} FR:{}".format(
            current_stats["captures"], current_stats["frames"]
        ), (255, 255, 255))

        if capture_remaining > 0:
            draw_label(osd_img, 8, 80, "CAPTURING {} {}/{}".format(
                capture_color,
                CAPTURE_FRAME_COUNT - capture_remaining,
                CAPTURE_FRAME_COUNT,
            ), (255, 0, 255))
        else:
            draw_label(osd_img, 8, 80, "C:CAPTURE P:PRINT H:HELP", (255, 255, 0))

        draw_label(osd_img, 8, 104, "KEY:{} CV:{}ms".format(last_key_text, process_ms), (0, 255, 0))
        pl.show_image()

        key = cv2.waitKey(1)
        if key is not None and key >= 0:
            key_code = key & 0xFF
            try:
                key_char = chr(key_code)
            except Exception:
                key_char = ""
            last_key_text = key_char

            # 采样中锁定 ROI 与颜色，避免一组样本中途改变位置或标签。
            if capture_remaining > 0:
                if key_char in ("q", "Q"):
                    should_quit = True
            elif key_char in ("r", "R"):
                current_color = "RED"
            elif key_char in ("b", "B"):
                current_color = "BLUE"
            elif key_char in ("y", "Y"):
                current_color = "YELLOW"
            elif key_char in ("k", "K"):
                current_color = "BLACK"
            elif key_char in ("w", "W"):
                roi_cy = clamp(roi_cy - ROI_MOVE_STEP, roi_size // 2, CV_SIZE[1] - roi_size // 2)
            elif key_char in ("s", "S"):
                roi_cy = clamp(roi_cy + ROI_MOVE_STEP, roi_size // 2, CV_SIZE[1] - roi_size // 2)
            elif key_char in ("a", "A"):
                roi_cx = clamp(roi_cx - ROI_MOVE_STEP, roi_size // 2, CV_SIZE[0] - roi_size // 2)
            elif key_char in ("d", "D"):
                roi_cx = clamp(roi_cx + ROI_MOVE_STEP, roi_size // 2, CV_SIZE[0] - roi_size // 2)
            elif key_char in ("+", "="):
                roi_size = clamp(roi_size + ROI_SIZE_STEP, ROI_MIN_SIZE, ROI_MAX_SIZE)
                roi_cx = clamp(roi_cx, roi_size // 2, CV_SIZE[0] - roi_size // 2)
                roi_cy = clamp(roi_cy, roi_size // 2, CV_SIZE[1] - roi_size // 2)
            elif key_char in ("-", "_"):
                roi_size = clamp(roi_size - ROI_SIZE_STEP, ROI_MIN_SIZE, ROI_MAX_SIZE)
            elif key_char in ("c", "C"):
                capture_color = current_color
                capture_remaining = CAPTURE_FRAME_COUNT
                print(">>> start capture: {} ROI=({}, {}, {}, {})".format(
                    capture_color, roi_x, roi_y, roi_w, roi_h
                ))
            elif key_char in ("x", "X"):
                reset_color_stats(all_stats, current_color)
                print(">>> cleared samples:", current_color)
            elif key_char in ("p", "P"):
                print_and_save_results(all_stats)
            elif key_char in ("h", "H"):
                print_help()
            elif key_char in ("q", "Q"):
                should_quit = True

        del frame_hwc, small_rgb, hsv, hsv_roi, hue_valid_mask
        del hist_h, hist_s, hist_v

        if frame_count % 30 == 0:
            gc.collect()
        time.sleep_ms(5)

except KeyboardInterrupt as e:
    print("user stopped:", e)
except BaseException as e:
    if str(e) == "IDE interrupt":
        print("user stopped by IDE")
    else:
        print("interactive HSV calibration FAILED:", e)
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
    print("interactive HSV calibration finished")
