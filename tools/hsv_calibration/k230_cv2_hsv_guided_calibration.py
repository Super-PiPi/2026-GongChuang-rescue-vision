# ============================================================
# K230 OpenCV 屏幕引导式 HSV 快速标定器（无需键盘）
# 自动引导红、蓝、黄、黑目标完成近处、中距离、画面边缘三组采样。
# 最终 HSV_RANGES 打印到串口并保存到 /sdcard/hsv_calibration.txt。
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


# ===================== 现场可快速调整的少量参数 =====================
SENSOR_ID = 1
PIPELINE_SIZE = [800, 480]
CV_SIZE = [320, 240]
DISPLAY_MODE = "lcd"
SAVE_PATH = "/sdcard/hsv_calibration.txt"

# 每一步给操作者放置目标的时间，以及实际累计采样帧数。
PLACE_TIME_MS = 3500
CAPTURE_FRAME_COUNT = 10
STEP_DONE_TIME_MS = 800
START_READY_TIME_MS = 4000

# 以下 ROI 坐标/尺寸基于 320x240 CV 图像。
# NEAR 框较大；MID 和 EDGE 框较小，适合中远距离小目标。
NEAR_ROI_SIZE = 24
MID_ROI_SIZE = 8
EDGE_ROI_SIZE = 8
EDGE_ROI_CENTER_X = 32

# 统计范围之外额外放宽的 HSV 余量。
H_MARGIN = 3
S_MARGIN = 15
V_MARGIN = 15
BLACK_V_MARGIN = 10
HUE_VALID_MIN_S = 40
HUE_VALID_MIN_V = 40

COLOR_NAMES = ("RED", "BLUE", "YELLOW", "BLACK")
OSD_COLORS = {
    "RED": (255, 0, 0),
    "BLUE": (0, 0, 255),
    "YELLOW": (255, 255, 0),
    "BLACK": (255, 255, 255),
}

# 一次运行自动执行全部步骤。EDGE 位于画面左边缘，可在上方修改 X。
CALIBRATION_STEPS = (
    ("RED", "NEAR", CV_SIZE[0] // 2, CV_SIZE[1] // 2, NEAR_ROI_SIZE),
    ("RED", "MID", CV_SIZE[0] // 2, CV_SIZE[1] // 2, MID_ROI_SIZE),
    ("RED", "EDGE", EDGE_ROI_CENTER_X, CV_SIZE[1] // 2, EDGE_ROI_SIZE),
    ("BLUE", "NEAR", CV_SIZE[0] // 2, CV_SIZE[1] // 2, NEAR_ROI_SIZE),
    ("BLUE", "MID", CV_SIZE[0] // 2, CV_SIZE[1] // 2, MID_ROI_SIZE),
    ("BLUE", "EDGE", EDGE_ROI_CENTER_X, CV_SIZE[1] // 2, EDGE_ROI_SIZE),
    ("YELLOW", "NEAR", CV_SIZE[0] // 2, CV_SIZE[1] // 2, NEAR_ROI_SIZE),
    ("YELLOW", "MID", CV_SIZE[0] // 2, CV_SIZE[1] // 2, MID_ROI_SIZE),
    ("YELLOW", "EDGE", EDGE_ROI_CENTER_X, CV_SIZE[1] // 2, EDGE_ROI_SIZE),
    ("BLACK", "NEAR", CV_SIZE[0] // 2, CV_SIZE[1] // 2, NEAR_ROI_SIZE),
    ("BLACK", "MID", CV_SIZE[0] // 2, CV_SIZE[1] // 2, MID_ROI_SIZE),
    ("BLACK", "EDGE", EDGE_ROI_CENTER_X, CV_SIZE[1] // 2, EDGE_ROI_SIZE),
)


def draw_label(img, x, y, text, color=(0, 255, 0), char_size=24):
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
        "frames": 0,
        "captures": 0,
    }


def hist_value(hist, index):
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
    return [(0, expanded_end_unwrapped - 180), (expanded_start, 179)]


def add_histogram(sum_hist, new_hist):
    if sum_hist is None:
        return new_hist
    return cv2.add(sum_hist, new_hist)


def get_cv_roi(step):
    _, _, center_x, center_y, size = step
    half = size // 2
    x = clamp(center_x - half, 0, CV_SIZE[0] - size)
    y = clamp(center_y - half, 0, CV_SIZE[1] - size)
    return x, y, size, size


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


def add_sample_frame(stats, hsv_roi, hue_mask_low, hue_mask_high):
    hue_mask = cv2.inRange(hsv_roi, hue_mask_low, hue_mask_high)
    hist_h = cv2.calcHist([hsv_roi], [0], hue_mask, [180], [0, 180])
    hist_s = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv_roi], [2], None, [256], [0, 256])

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
    stats["frames"] += 1

    live_h = histogram_percentile(hist_h, 180, 0.50)
    live_s = histogram_percentile(hist_s, 256, 0.50)
    live_v = histogram_percentile(hist_v, 256, 0.50)

    del hue_mask, hist_h, hist_s, hist_v, channels
    return live_h, live_s, live_v


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
    lines.append("K230 GUIDED HSV CALIBRATION RESULT")
    lines.append("HSV_RANGES = {")
    for color_name in COLOR_NAMES:
        stats = all_stats[color_name]
        result = summarize_color(color_name, stats)
        if result is None:
            lines.append('    # "{}": no samples'.format(color_name))
            continue

        lines.append("    # {} captures={} frames={}".format(
            color_name, stats["captures"], stats["frames"]
        ))
        lines.append("    # raw min={} raw max={}".format(stats["raw_min"], stats["raw_max"]))
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


PHASE_READY = 0
PHASE_PLACE = 1
PHASE_CAPTURE = 2
PHASE_DONE = 3
PHASE_COMPLETE = 4

pl = None
sensor = None

try:
    os.exitpoint(os.EXITPOINT_ENABLE)
    print("===== K230 guided HSV calibration (no keyboard) =====")
    print("steps:{} place_time:{}ms capture_frames:{}".format(
        len(CALIBRATION_STEPS), PLACE_TIME_MS, CAPTURE_FRAME_COUNT
    ))

    required_apis = (
        "resize", "cvtColor", "calcHist", "inRange",
        "split", "minMaxLoc", "add",
    )
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

    all_stats = {}
    for color_name in COLOR_NAMES:
        all_stats[color_name] = new_color_stats()

    hue_mask_low = np.array([0, HUE_VALID_MIN_S, HUE_VALID_MIN_V], dtype=np.uint8)
    hue_mask_high = np.array([179, 255, 255], dtype=np.uint8)

    phase = PHASE_READY
    phase_start_ms = time.ticks_ms()
    step_index = 0
    capture_count = 0
    live_hsv = (0, 0, 0)
    results_written = False

    while True:
        os.exitpoint()
        frame_chw = pl.get_frame()
        osd_img = pl.osd_img
        osd_img.clear()
        now_ms = time.ticks_ms()
        phase_elapsed = time.ticks_diff(now_ms, phase_start_ms)

        if phase == PHASE_READY:
            remain_ms = START_READY_TIME_MS - phase_elapsed
            draw_label(osd_img, 8, 8, "HSV GUIDED CALIBRATION", (0, 255, 0), 28)
            draw_label(osd_img, 8, 46, "PREPARE TARGETS", (255, 255, 255), 26)
            draw_label(osd_img, 8, 82, "START IN {:.1f}s".format(max(0, remain_ms) / 1000), (255, 255, 0), 26)
            if phase_elapsed >= START_READY_TIME_MS:
                phase = PHASE_PLACE
                phase_start_ms = now_ms

        elif phase == PHASE_COMPLETE:
            draw_label(osd_img, 8, 8, "CALIBRATION COMPLETE", (0, 255, 0), 28)
            draw_label(osd_img, 8, 46, "CHECK SERIAL OUTPUT", (255, 255, 255), 24)
            draw_label(osd_img, 8, 78, "SAVED: /sdcard/", (0, 255, 255), 22)
            draw_label(osd_img, 8, 106, "hsv_calibration.txt", (0, 255, 255), 22)
            if not results_written:
                print_and_save_results(all_stats)
                results_written = True
                gc.collect()

        else:
            step = CALIBRATION_STEPS[step_index]
            color_name, scene_name, _, _, _ = step
            cv_roi = get_cv_roi(step)
            display_roi = cv_roi_to_display(cv_roi, display_size)
            roi_color = OSD_COLORS[color_name]
            osd_img.draw_rectangle(display_roi, color=roi_color, thickness=3)
            osd_img.draw_cross(
                display_roi[0] + display_roi[2] // 2,
                display_roi[1] + display_roi[3] // 2,
                color=roi_color,
                size=10,
            )

            draw_label(osd_img, 8, 8, "STEP:{}/{} {}".format(
                step_index + 1, len(CALIBRATION_STEPS), color_name
            ), roi_color, 26)
            draw_label(osd_img, 8, 42, "SCENE:{} ROI:{}".format(
                scene_name, cv_roi[2]
            ), (255, 255, 255), 24)

            if phase == PHASE_PLACE:
                remain_ms = PLACE_TIME_MS - phase_elapsed
                draw_label(osd_img, 8, 74, "PLACE OBJECT IN BOX", (255, 255, 0), 24)
                draw_label(osd_img, 8, 106, "CAPTURE IN {:.1f}s".format(
                    max(0, remain_ms) / 1000
                ), (255, 255, 0), 24)
                if phase_elapsed >= PLACE_TIME_MS:
                    phase = PHASE_CAPTURE
                    phase_start_ms = now_ms
                    capture_count = 0
                    print(">>> capture step {}/{}: {} {} ROI={}".format(
                        step_index + 1, len(CALIBRATION_STEPS), color_name, scene_name, cv_roi
                    ))

            elif phase == PHASE_CAPTURE:
                frame_hwc = chw2hwc(frame_chw)
                small_rgb = cv2.resize(
                    frame_hwc,
                    (CV_SIZE[0], CV_SIZE[1]),
                    interpolation=cv2.INTER_AREA,
                )
                hsv = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
                roi_x, roi_y, roi_w, roi_h = cv_roi
                hsv_roi = hsv[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, :]
                live_hsv = add_sample_frame(
                    all_stats[color_name], hsv_roi, hue_mask_low, hue_mask_high
                )
                capture_count += 1

                draw_label(osd_img, 8, 74, "CAPTURING:{}/{}".format(
                    capture_count, CAPTURE_FRAME_COUNT
                ), (255, 0, 255), 24)
                draw_label(osd_img, 8, 106, "H:{} S:{} V:{}".format(
                    live_hsv[0], live_hsv[1], live_hsv[2]
                ), (0, 255, 255), 24)

                del frame_hwc, small_rgb, hsv, hsv_roi

                if capture_count >= CAPTURE_FRAME_COUNT:
                    all_stats[color_name]["captures"] += 1
                    phase = PHASE_DONE
                    phase_start_ms = now_ms
                    print(">>> captured: {} {} live_hsv={} total_frames={}".format(
                        color_name,
                        scene_name,
                        live_hsv,
                        all_stats[color_name]["frames"],
                    ))
                    gc.collect()

            elif phase == PHASE_DONE:
                draw_label(osd_img, 8, 74, "CAPTURED", (0, 255, 0), 28)
                draw_label(osd_img, 8, 110, "NEXT STEP...", (255, 255, 255), 24)
                if phase_elapsed >= STEP_DONE_TIME_MS:
                    step_index += 1
                    if step_index >= len(CALIBRATION_STEPS):
                        phase = PHASE_COMPLETE
                    else:
                        phase = PHASE_PLACE
                    phase_start_ms = now_ms

        pl.show_image()
        time.sleep_ms(10)

except KeyboardInterrupt as e:
    print("user stopped:", e)
except BaseException as e:
    if str(e) == "IDE interrupt":
        print("user stopped by IDE")
    else:
        print("guided HSV calibration FAILED:", e)
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
    print("guided HSV calibration finished")

