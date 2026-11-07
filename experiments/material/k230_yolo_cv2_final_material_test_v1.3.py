# ============================================================
# K230 决赛四类物料 YOLO + OpenCV HSV 独立识别测试 v1.3
# YOLO：红球、蓝球、安全区；OpenCV：黑色正方体、黄色三棱锥。
# 仅检测、融合、显示和打印，不初始化 UART，不控制底盘与夹爪。
# V1.3：根据实测提高近场门限，加入空夹爪基线标定、连续确认和防回跳匹配。
# ============================================================

import gc
import os
import time
import image
import cv2

from ulab import numpy as np
from media.sensor import *
from media.display import *
from media.media import MediaManager
from libs.YOLO import YOLOv5


VERSION = "K230 FINAL YOLO+CV2 material test v1.3"

# ===================== camera / display =====================
SENSOR_ID = 1
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480
YOLO_WIDTH = 800
YOLO_HEIGHT = 480
CV_WIDTH = 320
CV_HEIGHT = 240

# CH0：LCD/IDE 背景；CH1：OpenCV 320x240 RGB888/HWC；CH2：YOLO RGBP888/CHW。
# RGB888 的 to_numpy_ref() 直接返回 HWC，不再执行 800x480 CHW->HWC 和 resize。
DISPLAY_CHN = CAM_CHN_ID_0
CV_CHN = CAM_CHN_ID_1
YOLO_CHN = CAM_CHN_ID_2

# OpenCV 每隔多少帧处理一次。1=每帧；若整机帧率太低可先改为 2。
CV_PROCESS_EVERY_N_FRAMES = 1

# ===================== CV near-gripper lock test =====================
# 本段只做识别诊断，不会触发真实抓取。测试时请把目标从远处缓慢移入夹爪。
# 近夹爪 ROI 使用 800x480 显示坐标；内部会自动换算为 320x240 CV 坐标。
NEAR_GRIP_ROI = (300, 240, 240, 240)

# 严格 CV 目标连续出现 3 帧后自动锁定，避免单帧阴影误检直接进入近场通路。
AUTO_LOCK_CONFIRM_FRAMES = 3
STRICT_MATCH_MAX_DISTANCE = 180
STRICT_MATCH_MAX_X_DELTA = 120
NEAR_STRICT_MAX_BACKTRACK_Y = 80

# 上电后的前 20 帧自动记录空夹爪 ROI 的黑/黄像素基线。
# 运行脚本时目标必须先放在紫色 ROI 外；标定完成后会打印 EMPTY_BASELINES。
EMPTY_BASELINE_CALIBRATION_FRAMES = 20

# 宽松近场证据也必须连续成立，避免单帧阴影或曝光波动刷新锁定计时。
NEAR_CONFIRM_FRAMES = 2

# 严格轮廓和近场色块都消失后，短时间保留“刚刚确实到达夹爪”的证据。
# LATCH 只用于本测试观察遮挡过程；3 秒持续丢失后自动解锁，便于换下一种目标。
NEAR_LATCH_MS = 900
LOCK_RELEASE_MS = 2200
DEBUG_PRINT_EVERY_N_FRAMES = 15

# enter_aim_y：只有目标已经足够靠近，才允许进入 NEAR/LATCH。
# min_excess_pixels：相对锁定时的空夹爪基线，ROI 至少新增多少个同色像素。
NEAR_TRACK_CONFIG = {
    "CORE": {
        "enter_aim_y": 180,
        # 实测空夹爪伪轮廓 A≈555、EX≈25~34；真黑方块进入夹爪后
        # A≈1800~3000、EX≈1800~2200，因此保留足够安全余量。
        "min_contour_area": 700,
        "min_excess_pixels": 150,
        "max_x_delta": 100,
    },
    "DANGER": {
        "enter_aim_y": 300,
        # 实测真黄色近场 A≈1720、EX≈1080~1130。
        "min_contour_area": 150,
        "min_excess_pixels": 80,
        "max_x_delta": 100,
    },
}

# 是否显示被几何条件拒绝的轮廓；现场调参阶段建议保持 True。
SHOW_REJECTED_CV = True
MAX_REJECTED_DRAW = 12

# ===================== model =====================
KMODEL_PATH = "/sdcard/model/best_0719.kmodel"
MODEL_INPUT_SIZE = [320, 320]
LABELS = [
    "redball",
    "blueball",
    "redboard",
    "blueboard",
    "redsection",
    "bluesection",
    "shizi",
    "redcube",
    "bluecube",
    "greenzhu",
    "yellowtai",
    "blackball",
    "yellowball",
]
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45
MAX_BOXES_NUM = 50

YOLO_NORMAL_BALL_LABELS = ("redball", "blueball")
YOLO_SAFE_LABELS = ("redboard", "blueboard")

# 决赛不存在黑球和黄球。模型的这两个输出只给同色 HSV 候选加辅助标记，
# 不能脱离 HSV 候选独立成为黑方块/黄三棱锥控制目标。
YOLO_ASSIST_LABELS = ("blackball", "yellowball")

# 只影响顶部 PRIMARY 选择；红、蓝球都会被显示。
TEAM_COLOR = "BLUE"  # 现场改成 "RED" 或 "BLUE"

# ===================== OpenCV HSV thresholds =====================
# OpenCV HSV 范围：H=0..179，S/V=0..255。
# 黑色主要调整 V 最大值，H 对黑色没有区分意义。
BLACK_HSV_RANGES = (
    ((0, 0, 0), (179, 255, 70)),
)

# 黄色现场主要调整 H 范围、S 最小值和 V 最小值。
YELLOW_HSV_RANGES = (
    ((18, 80, 60), (38, 255, 255)),
)

# 所有面积和尺寸均基于 320x240 CV 图像。
# 参数集中在两个字典中，接入主状态机时可以直接复用。
CV_MATERIAL_CONFIG = {
    "CORE": {
        "color": "BLACK",
        "ranges": BLACK_HSV_RANGES,
        "draw_color": (255, 255, 255),
        "min_contour_area": 60,
        "max_contour_area": 22000,
        "min_width": 7,
        "min_height": 7,
        # 黑方块允许一定透视和侧面可见，但拒绝明显细长暗线。
        "min_aspect": 0.55,
        "max_aspect": 1.80,
        "min_fill": 0.48,
        "max_fill": 1.05,
        # 外扩区域平均 V - 框内平均 V。白色场地上的真黑方块通常明显大于阴影。
        "min_local_contrast": 10.0,
        "use_local_contrast": True,
        # 黑方块不做强闭运算，避免多个相邻方块粘成一个大轮廓。
        "use_close": False,
        "aim_y_ratio": 0.78,
    },
    "DANGER": {
        "color": "YELLOW",
        "ranges": YELLOW_HSV_RANGES,
        "draw_color": (255, 255, 0),
        "min_contour_area": 60,
        "max_contour_area": 26000,
        "min_width": 7,
        "min_height": 7,
        # 三棱锥视角变化大，不强制轮廓必须是三角形。
        "min_aspect": 0.45,
        "max_aspect": 2.20,
        "min_fill": 0.28,
        "max_fill": 1.05,
        "min_local_contrast": 0.0,
        "use_local_contrast": False,
        "use_close": True,
        "aim_y_ratio": 0.82,
    },
}

CV_MATERIAL_ORDER = ("CORE", "DANGER")

# 只排除恰好贴住图像边缘的轮廓，避免把画面外的大块黑色区域截成目标。
CV_BORDER_MARGIN = 2

# 固定车体/夹爪遮挡区，可在实机画面确认后填入 320x240 坐标矩形。
# 格式：(x, y, width, height)。当前为空，不等同于“地面有效区”。
CV_EXCLUDE_RECTS = ()

# HSV 与 YOLO 来自同一帧，但框大小仍不同。
YOLO_OVERLAP_MARGIN = 12
CV_COVERED_RATIO_MIN = 0.20

# ===================== draw colors =====================
COLOR_YOLO_OWN = (0, 255, 0)
COLOR_YOLO_OPP = (100, 100, 100)
COLOR_YOLO_SAFE = (255, 0, 255)
COLOR_YOLO_ASSIST = (0, 180, 255)
COLOR_CV_CORE = (255, 255, 255)
COLOR_CV_DANGER = (255, 255, 0)
COLOR_CV_REJECT = (255, 128, 0)
COLOR_CV_DROP_BALL = (120, 120, 120)
COLOR_CV_DROP_SAFE = (255, 0, 128)
COLOR_PRIMARY = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_NEAR_ROI = (180, 0, 255)
COLOR_NEAR_OK = (0, 255, 120)
COLOR_NEAR_WEAK = (255, 80, 180)
COLOR_LOCKED = (0, 180, 255)


def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def free_memory_text():
    try:
        return str(gc.mem_free())
    except Exception:
        return "N/A"


def draw_text(img, x, y, text, color=COLOR_TEXT):
    x = clamp(int(x), 0, DISPLAY_WIDTH - 8)
    y = clamp(int(y), 0, DISPLAY_HEIGHT - 12)
    try:
        # draw_string 在当前固件会每次打印 deprecated 警告，严重污染串口日志。
        img.draw_string_advanced(x, y, 12, text, color)
    except Exception:
        pass


def check_cv2_apis():
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
        "mean",
    )
    missing = []
    for name in required:
        if not hasattr(cv2, name):
            missing.append(name)
    if missing:
        raise RuntimeError("current firmware lacks cv2 APIs: {}".format(missing))

    optional = ("arcLength", "approxPolyDP", "isContourConvex")
    available_optional = []
    for name in optional:
        if hasattr(cv2, name):
            available_optional.append(name)
    print("cv2 required APIs: PASS")
    print("cv2 optional shape APIs:", available_optional)


def parse_yolo_results(res):
    """兼容当前 K230 YOLOv5 返回值，并按 xywh 正确转换检测框。"""
    detections = []
    if res is None:
        return detections

    def append_det(x1, y1, x2, y2, score, class_id, label):
        if label is None and class_id is not None:
            try:
                class_index = int(class_id)
                if 0 <= class_index < len(LABELS):
                    label = LABELS[class_index]
            except Exception:
                label = None

        try:
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
        except Exception:
            return

        if x2 < x1 or y2 < y1:
            x2 = x1 + max(0, x2)
            y2 = y1 + max(0, y2)

        x1 = clamp(x1, 0, YOLO_WIDTH - 1)
        y1 = clamp(y1, 0, YOLO_HEIGHT - 1)
        x2 = clamp(x2, 0, YOLO_WIDTH)
        y2 = clamp(y2, 0, YOLO_HEIGHT)
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return

        if score is None:
            score = 0.0

        detections.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "w": width,
            "h": height,
            "cx": x1 + width // 2,
            "cy": y1 + height // 2,
            "area": width * height,
            "score": float(score),
            "class_id": class_id,
            "label": label,
            "rect": (x1, y1, width, height),
        })

    # 当前 K230 aidemo.yolov5_det_postprocess 标准输出：
    # (boxes, class_ids, scores)，box 为 [x, y, width, height]。
    if isinstance(res, (tuple, list)) and len(res) == 3:
        boxes = res[0]
        class_ids = res[1]
        scores = res[2]
        if isinstance(boxes, (tuple, list)) and isinstance(class_ids, (tuple, list)):
            count = min(len(boxes), len(class_ids))
            if isinstance(scores, (tuple, list)):
                count = min(count, len(scores))
            for index in range(count):
                box = boxes[index]
                if box is None or len(box) < 4:
                    continue
                score = scores[index] if isinstance(scores, (tuple, list)) else 0.0
                append_det(
                    box[0],
                    box[1],
                    box[0] + box[2],
                    box[1] + box[3],
                    score,
                    class_ids[index],
                    None,
                )
            return detections

    # 保留字典、列表和对象格式兼容。
    for entry in res:
        if entry is None:
            continue

        x1 = y1 = x2 = y2 = score = class_id = label = None
        if isinstance(entry, dict):
            x1 = entry.get("x1") if entry.get("x1") is not None else entry.get("x")
            y1 = entry.get("y1") if entry.get("y1") is not None else entry.get("y")
            x2 = entry.get("x2")
            y2 = entry.get("y2")
            score = entry.get("score")
            if score is None:
                score = entry.get("confidence")
            if score is None:
                score = entry.get("prob")
            class_id = entry.get("class_id")
            if class_id is None:
                class_id = entry.get("classid")
            if class_id is None:
                class_id = entry.get("classId")
            label = entry.get("label")
            if x2 is None and entry.get("w") is not None and x1 is not None:
                x2 = x1 + entry.get("w")
            if y2 is None and entry.get("h") is not None and y1 is not None:
                y2 = y1 + entry.get("h")
        elif isinstance(entry, (tuple, list)):
            if len(entry) >= 6:
                a0, a1, a2, a3, a4, a5 = entry[:6]
                if int(a0) >= 0 and int(a0) < len(LABELS) and float(a1) <= 1.0:
                    class_id, score, x1, y1, x2, y2 = a0, a1, a2, a3, a4, a5
                else:
                    x1, y1, x2, y2, score, class_id = a0, a1, a2, a3, a4, a5
            elif len(entry) == 5:
                x1, y1, x2, y2, score = entry
            else:
                continue
        else:
            x1 = getattr(entry, "x1", None)
            if x1 is None:
                x1 = getattr(entry, "x", None)
            y1 = getattr(entry, "y1", None)
            if y1 is None:
                y1 = getattr(entry, "y", None)
            x2 = getattr(entry, "x2", None)
            y2 = getattr(entry, "y2", None)
            score = getattr(entry, "score", None)
            if score is None:
                score = getattr(entry, "confidence", None)
            class_id = getattr(entry, "class_id", None)
            if class_id is None:
                class_id = getattr(entry, "classid", None)
            label = getattr(entry, "label", None)
            if x2 is None and getattr(entry, "w", None) is not None and x1 is not None:
                x2 = x1 + getattr(entry, "w")
            if y2 is None and getattr(entry, "h", None) is not None and y1 is not None:
                y2 = y1 + getattr(entry, "h")

        append_det(x1, y1, x2, y2, score, class_id, label)

    return detections


def compile_hsv_ranges():
    compiled = {}
    for target_type in CV_MATERIAL_ORDER:
        pairs = []
        config = CV_MATERIAL_CONFIG[target_type]
        for lower, upper in config["ranges"]:
            pairs.append((
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            ))
        compiled[target_type] = pairs
    return compiled


def build_mask(hsv_img, threshold_pairs):
    mask = None
    for lower, upper in threshold_pairs:
        part = cv2.inRange(hsv_img, lower, upper)
        if mask is None:
            mask = part
        else:
            merged = cv2.bitwise_or(mask, part)
            del mask
            del part
            mask = merged
    return mask


def get_contours(mask):
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 2:
        return result[0]
    return result[1]


def rect_touches_exclusion(x, y, width, height):
    cx = x + width // 2
    cy = y + height // 2
    for ex, ey, ew, eh in CV_EXCLUDE_RECTS:
        if ex <= cx <= ex + ew and ey <= cy <= ey + eh:
            return True
    return False


def get_local_v_contrast(hsv_img, x, y, width, height):
    # 使用比目标框每边多约半个框宽/高的外扩区域。
    pad_x = max(4, width // 2)
    pad_y = max(4, height // 2)
    ox1 = clamp(x - pad_x, 0, CV_WIDTH - 1)
    oy1 = clamp(y - pad_y, 0, CV_HEIGHT - 1)
    ox2 = clamp(x + width + pad_x, ox1 + 1, CV_WIDTH)
    oy2 = clamp(y + height + pad_y, oy1 + 1, CV_HEIGHT)

    inner_roi = hsv_img[y:y + height, x:x + width, :]
    outer_roi = hsv_img[oy1:oy2, ox1:ox2, :]
    inner_v = float(cv2.mean(inner_roi)[2])
    outer_v = float(cv2.mean(outer_roi)[2])
    return outer_v - inner_v, inner_v, outer_v


def optional_vertex_count(contour):
    if not hasattr(cv2, "arcLength") or not hasattr(cv2, "approxPolyDP"):
        return -1
    try:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        return len(approximation)
    except Exception:
        return -1


def evaluate_contour(hsv_img, contour, target_type, config):
    contour_area = float(cv2.contourArea(contour))
    x, y, width, height = cv2.boundingRect(contour)
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    box_area = max(1, width * height)
    aspect = float(width) / float(max(1, height))
    fill = contour_area / float(box_area)
    contrast = 0.0
    inner_v = 0.0
    outer_v = 0.0
    vertices = optional_vertex_count(contour)

    metrics = {
        "contour_area_320": contour_area,
        "box_area_320": box_area,
        "aspect": aspect,
        "fill": fill,
        "contrast": contrast,
        "inner_v": inner_v,
        "outer_v": outer_v,
        "vertices": vertices,
        "cv_rect": (x, y, width, height),
    }

    if contour_area < config["min_contour_area"]:
        return False, "AREA_LOW", metrics
    if contour_area > config["max_contour_area"]:
        return False, "AREA_HIGH", metrics
    if width < config["min_width"] or height < config["min_height"]:
        return False, "SIZE", metrics
    if x <= CV_BORDER_MARGIN or y <= CV_BORDER_MARGIN:
        return False, "BORDER", metrics
    if x + width >= CV_WIDTH - CV_BORDER_MARGIN:
        return False, "BORDER", metrics
    if y + height >= CV_HEIGHT - CV_BORDER_MARGIN:
        return False, "BORDER", metrics
    if rect_touches_exclusion(x, y, width, height):
        return False, "SELF", metrics
    if aspect < config["min_aspect"] or aspect > config["max_aspect"]:
        return False, "ASPECT", metrics
    if fill < config["min_fill"] or fill > config["max_fill"]:
        return False, "FILL", metrics

    if config["use_local_contrast"]:
        contrast, inner_v, outer_v = get_local_v_contrast(
            hsv_img, x, y, width, height
        )
        metrics["contrast"] = contrast
        metrics["inner_v"] = inner_v
        metrics["outer_v"] = outer_v
        if contrast < config["min_local_contrast"]:
            return False, "CONTRAST", metrics

    # 顶点数只记录，不作为硬门禁。真方块在透视、高光和遮挡下不一定严格为四边形。
    return True, "OK", metrics


def map_cv_metrics_to_target(target_type, config, metrics):
    x, y, width, height = metrics["cv_rect"]
    x1 = int(x * YOLO_WIDTH / CV_WIDTH)
    y1 = int(y * YOLO_HEIGHT / CV_HEIGHT)
    mapped_width = max(1, int(width * YOLO_WIDTH / CV_WIDTH))
    mapped_height = max(1, int(height * YOLO_HEIGHT / CV_HEIGHT))
    x2 = clamp(x1 + mapped_width, 0, YOLO_WIDTH)
    y2 = clamp(y1 + mapped_height, 0, YOLO_HEIGHT)
    x1 = clamp(x1, 0, YOLO_WIDTH - 1)
    y1 = clamp(y1, 0, YOLO_HEIGHT - 1)
    mapped_width = max(1, x2 - x1)
    mapped_height = max(1, y2 - y1)
    cx = x1 + mapped_width // 2
    cy = y1 + mapped_height // 2
    aim_y = y1 + int(mapped_height * config["aim_y_ratio"])
    if aim_y >= y2:
        aim_y = y2 - 1

    target = {
        "source": "CV2",
        "type": target_type,
        "color": config["color"],
        "verify_key": config["color"],
        "draw_color": config["draw_color"],
        "rect": (x1, y1, mapped_width, mapped_height),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "w": mapped_width,
        "h": mapped_height,
        "cx": cx,
        "cy": cy,
        "aim_x": cx,
        "aim_y": aim_y,
        "area": mapped_width * mapped_height,
        "contour_area_320": metrics["contour_area_320"],
        "box_area_320": metrics["box_area_320"],
        "aspect": metrics["aspect"],
        "fill": metrics["fill"],
        "contrast": metrics["contrast"],
        "inner_v": metrics["inner_v"],
        "outer_v": metrics["outer_v"],
        "vertices": metrics["vertices"],
        "yolo_support": False,
        "drop_reason": None,
    }
    return target


def collect_cv_materials(hsv_img, compiled_ranges, open_kernel, close_kernel):
    accepted = []
    rejected = []
    raw_counts = {}
    reject_counts = {}

    for target_type in CV_MATERIAL_ORDER:
        config = CV_MATERIAL_CONFIG[target_type]
        mask = build_mask(hsv_img, compiled_ranges[target_type])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        if config["use_close"]:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

        contours = get_contours(mask)
        raw_counts[target_type] = len(contours)

        for contour in contours:
            valid, reason, metrics = evaluate_contour(
                hsv_img, contour, target_type, config
            )
            target = map_cv_metrics_to_target(target_type, config, metrics)
            if valid:
                accepted.append(target)
            else:
                key = "{}:{}".format(target_type, reason)
                reject_counts[key] = reject_counts.get(key, 0) + 1
                # 很小的噪点只计数，不画框，避免 OSD 被淹没。
                if metrics["contour_area_320"] >= config["min_contour_area"]:
                    target["drop_reason"] = reason
                    rejected.append(target)

        del contours
        del mask

    rejected.sort(key=lambda item: item["contour_area_320"], reverse=True)
    if len(rejected) > MAX_REJECTED_DRAW:
        rejected = rejected[:MAX_REJECTED_DRAW]
    return accepted, rejected, raw_counts, reject_counts


def boxes_overlap(cv_target, yolo_target):
    margin = YOLO_OVERLAP_MARGIN
    yx1 = clamp(yolo_target["x1"] - margin, 0, YOLO_WIDTH)
    yy1 = clamp(yolo_target["y1"] - margin, 0, YOLO_HEIGHT)
    yx2 = clamp(yolo_target["x2"] + margin, 0, YOLO_WIDTH)
    yy2 = clamp(yolo_target["y2"] + margin, 0, YOLO_HEIGHT)

    if yx1 <= cv_target["cx"] <= yx2 and yy1 <= cv_target["cy"] <= yy2:
        return True

    ix1 = max(cv_target["x1"], yx1)
    iy1 = max(cv_target["y1"], yy1)
    ix2 = min(cv_target["x2"], yx2)
    iy2 = min(cv_target["y2"], yy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False

    intersection = (ix2 - ix1) * (iy2 - iy1)
    return intersection / float(max(1, cv_target["area"])) >= CV_COVERED_RATIO_MIN


def fuse_cv_candidates(cv_candidates, yolo_normal_balls, yolo_safes, yolo_assists):
    accepted = []
    dropped_ball = []
    dropped_safe = []

    for target in cv_candidates:
        matched = None
        for yolo_target in yolo_normal_balls:
            if boxes_overlap(target, yolo_target):
                matched = yolo_target
                break
        if matched is not None:
            target["drop_reason"] = "BALL:{}".format(matched["label"])
            dropped_ball.append(target)
            continue

        for yolo_target in yolo_safes:
            if boxes_overlap(target, yolo_target):
                matched = yolo_target
                break
        if matched is not None:
            target["drop_reason"] = "SAFE:{}".format(matched["label"])
            dropped_safe.append(target)
            continue

        expected_assist = "blackball" if target["type"] == "CORE" else "yellowball"
        for yolo_target in yolo_assists:
            if yolo_target["label"] == expected_assist and boxes_overlap(target, yolo_target):
                target["yolo_support"] = True
                break

        accepted.append(target)

    return accepted, dropped_ball, dropped_safe


def choose_primary_target(yolo_normal_balls, cv_targets):
    own_label = "redball" if TEAM_COLOR == "RED" else "blueball"
    own_balls = []
    for target in yolo_normal_balls:
        if target["label"] == own_label:
            own_balls.append(target)
    if own_balls:
        return "YOLO_NORMAL", max(own_balls, key=lambda item: item["area"])

    # 根据实车抓取能力：黑方块优先于黄色三棱锥。
    for target_type in CV_MATERIAL_ORDER:
        candidates = []
        for target in cv_targets:
            if target["type"] == target_type:
                candidates.append(target)
        if candidates:
            return "CV2_{}".format(target_type), max(
                candidates,
                key=lambda item: (
                    1 if item["yolo_support"] else 0,
                    item["area"] - abs(item["cx"] - YOLO_WIDTH // 2) * 20,
                ),
            )

    return "NONE", None


def select_locked_strict_target(cv_targets, locked_type, last_x, last_y,
                                near_armed):
    """只在同类型严格候选中寻找与上一位置最接近的目标，禁止重新锁定别的物料。"""
    best = None
    best_distance_sq = None
    max_distance_sq = STRICT_MATCH_MAX_DISTANCE * STRICT_MATCH_MAX_DISTANCE

    for target in cv_targets:
        if target["type"] != locked_type:
            continue
        target_x = target.get("aim_x", target["cx"])
        target_y = target.get("aim_y", target["cy"])
        if last_x is None or last_y is None:
            distance_sq = 0
        else:
            dx = target_x - last_x
            dy = target_y - last_y
            distance_sq = dx * dx + dy * dy
            if abs(dx) > STRICT_MATCH_MAX_X_DELTA:
                continue
            # 目标进入夹爪后，只允许小幅向上抖动，拒绝跳到远处新出现的黑影。
            if near_armed and target_y < last_y - NEAR_STRICT_MAX_BACKTRACK_Y:
                continue
        if distance_sq > max_distance_sq:
            continue
        if best is None or distance_sq < best_distance_sq:
            best = target
            best_distance_sq = distance_sq

    return best


def display_rect_to_cv(rect):
    x, y, width, height = rect
    cv_x = clamp(int(x * CV_WIDTH / DISPLAY_WIDTH), 0, CV_WIDTH - 1)
    cv_y = clamp(int(y * CV_HEIGHT / DISPLAY_HEIGHT), 0, CV_HEIGHT - 1)
    cv_x2 = clamp(int((x + width) * CV_WIDTH / DISPLAY_WIDTH), cv_x + 1, CV_WIDTH)
    cv_y2 = clamp(int((y + height) * CV_HEIGHT / DISPLAY_HEIGHT), cv_y + 1, CV_HEIGHT)
    return cv_x, cv_y, cv_x2 - cv_x, cv_y2 - cv_y


def cv_rect_to_display(x, y, width, height):
    draw_x = int(x * DISPLAY_WIDTH / CV_WIDTH)
    draw_y = int(y * DISPLAY_HEIGHT / CV_HEIGHT)
    draw_width = max(1, int(width * DISPLAY_WIDTH / CV_WIDTH))
    draw_height = max(1, int(height * DISPLAY_HEIGHT / CV_HEIGHT))
    return draw_x, draw_y, draw_width, draw_height


def mask_pixel_count(mask):
    """用 cv2.mean 估算二值图中的白色像素数，兼容没有 countNonZero 的固件。"""
    if mask is None or len(mask.shape) < 2:
        return 0
    height = int(mask.shape[0])
    width = int(mask.shape[1])
    if width <= 0 or height <= 0:
        return 0
    mean_value = cv2.mean(mask)
    return int(float(mean_value[0]) * width * height / 255.0 + 0.5)


def measure_near_raw_pixels(hsv_img, target_type, compiled_ranges):
    """只统计近夹爪 ROI 的原始阈值像素，用于启动阶段记录空夹爪基线。"""
    cv_x, cv_y, cv_width, cv_height = display_rect_to_cv(NEAR_GRIP_ROI)
    full_mask = build_mask(hsv_img, compiled_ranges[target_type])
    roi_mask = full_mask[cv_y:cv_y + cv_height, cv_x:cv_x + cv_width]
    pixels = mask_pixel_count(roi_mask)
    del roi_mask
    del full_mask
    return pixels


def collect_near_grip_evidence(hsv_img, locked_type, compiled_ranges,
                               open_kernel, close_kernel, last_x):
    """在夹爪中央 ROI 内寻找同色的宽松轮廓，不使用边界、比例和填充率门禁。"""
    result = {
        "raw_pixels": 0,
        "contours": 0,
        "best_area": 0.0,
        "best_dx": 9999,
        "best_rect": None,
        "best_x": None,
        "best_y": None,
        "best_aim_y": None,
    }
    if locked_type not in NEAR_TRACK_CONFIG:
        return result

    cv_x, cv_y, cv_width, cv_height = display_rect_to_cv(NEAR_GRIP_ROI)
    full_mask = build_mask(hsv_img, compiled_ranges[locked_type])
    roi_mask = full_mask[cv_y:cv_y + cv_height, cv_x:cv_x + cv_width]
    result["raw_pixels"] = mask_pixel_count(roi_mask)

    processed = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, open_kernel)
    if CV_MATERIAL_CONFIG[locked_type]["use_close"]:
        closed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, close_kernel)
        del processed
        processed = closed

    contours = get_contours(processed)
    result["contours"] = len(contours)
    best_score = None
    max_x_delta = NEAR_TRACK_CONFIG[locked_type]["max_x_delta"]

    for contour in contours:
        area = float(cv2.contourArea(contour))
        local_x, local_y, width, height = cv2.boundingRect(contour)
        global_x = cv_x + int(local_x)
        global_y = cv_y + int(local_y)
        draw_rect = cv_rect_to_display(global_x, global_y, int(width), int(height))
        center_x = draw_rect[0] + draw_rect[2] // 2
        center_y = draw_rect[1] + draw_rect[3] // 2
        aim_y = draw_rect[1] + draw_rect[3] - 1
        dx = 0 if last_x is None else abs(center_x - last_x)
        if dx > max_x_delta:
            continue

        # 面积优先，同时轻微惩罚偏离上次锁定横坐标的轮廓。
        score = area - float(dx) * 0.25
        if best_score is None or score > best_score:
            best_score = score
            result["best_area"] = area
            result["best_dx"] = dx
            result["best_rect"] = draw_rect
            result["best_x"] = center_x
            result["best_y"] = center_y
            result["best_aim_y"] = aim_y

    del contours
    del processed
    del roi_mask
    del full_mask
    return result


def draw_results(osd_img, detections, yolo_normal_balls, yolo_safes,
                 yolo_assists, accepted_cv, rejected_cv,
                 dropped_ball, dropped_safe, primary_source,
                 primary_target, timing, raw_counts, reject_counts,
                 lock_debug):
    osd_img.clear()
    own_label = "redball" if TEAM_COLOR == "RED" else "blueball"

    for det in detections:
        if det["label"] in YOLO_NORMAL_BALL_LABELS:
            if det["label"] == own_label:
                color = COLOR_YOLO_OWN
                prefix = "YOLO-OWN"
                thickness = 3
            else:
                color = COLOR_YOLO_OPP
                prefix = "YOLO-OPP"
                thickness = 1
        elif det["label"] in YOLO_SAFE_LABELS:
            color = COLOR_YOLO_SAFE
            prefix = "YOLO-SAFE"
            thickness = 3
        elif det["label"] in YOLO_ASSIST_LABELS:
            color = COLOR_YOLO_ASSIST
            prefix = "YOLO-AUX"
            thickness = 1
        else:
            color = (80, 80, 80)
            prefix = "YOLO-OTHER"
            thickness = 1

        osd_img.draw_rectangle(det["rect"], color=color, thickness=thickness)
        draw_text(
            osd_img,
            det["x1"],
            det["y1"] - 12,
            "{} {} {:.2f}".format(prefix, det["label"], det["score"]),
            color,
        )

    if SHOW_REJECTED_CV:
        for target in rejected_cv:
            osd_img.draw_rectangle(target["rect"], color=COLOR_CV_REJECT, thickness=1)
            draw_text(
                osd_img,
                target["x1"],
                target["y1"] + target["h"],
                "REJ {} {}".format(target["type"], target["drop_reason"]),
                COLOR_CV_REJECT,
            )

    for target in dropped_ball:
        osd_img.draw_rectangle(target["rect"], color=COLOR_CV_DROP_BALL, thickness=1)
        draw_text(osd_img, target["x1"], target["y1"] + target["h"],
                  "DROP {}".format(target["drop_reason"]), COLOR_CV_DROP_BALL)

    for target in dropped_safe:
        osd_img.draw_rectangle(target["rect"], color=COLOR_CV_DROP_SAFE, thickness=1)
        draw_text(osd_img, target["x1"], target["y1"] + target["h"],
                  "DROP {}".format(target["drop_reason"]), COLOR_CV_DROP_SAFE)

    for target in accepted_cv:
        color = COLOR_CV_CORE if target["type"] == "CORE" else COLOR_CV_DANGER
        osd_img.draw_rectangle(target["rect"], color=color, thickness=3)
        osd_img.draw_cross(target["aim_x"], target["aim_y"], color=color, size=9)
        support_text = "+Y" if target["yolo_support"] else ""
        draw_text(
            osd_img,
            target["x1"],
            target["y1"] - 12,
            "CV2-{}{} A:{:.0f} F:{:.2f} C:{:.1f}".format(
                target["type"], support_text, target["contour_area_320"],
                target["fill"], target["contrast"]
            ),
            color,
        )

    if primary_target is not None:
        osd_img.draw_rectangle(primary_target["rect"], color=COLOR_PRIMARY, thickness=4)
        primary_x = primary_target.get("aim_x", primary_target["cx"])
        primary_y = primary_target.get("aim_y", primary_target["cy"])
        osd_img.draw_cross(primary_x, primary_y, color=COLOR_PRIMARY, size=13)

    # 近夹爪测试区不是地面有效区，只在目标已经自动锁定后提供同色原始像素证据。
    osd_img.draw_rectangle(NEAR_GRIP_ROI, color=COLOR_NEAR_ROI, thickness=2)
    draw_text(
        osd_img,
        NEAR_GRIP_ROI[0],
        NEAR_GRIP_ROI[1] - 14,
        "NEAR_GRIP_TEST_ROI",
        COLOR_NEAR_ROI,
    )

    locked_strict = lock_debug.get("strict_target")
    if locked_strict is not None:
        osd_img.draw_rectangle(locked_strict["rect"], color=COLOR_LOCKED, thickness=4)

    near_info = lock_debug.get("near_info")
    if near_info is not None and near_info.get("best_rect") is not None:
        near_color = COLOR_NEAR_OK if lock_debug.get("near_ok") else COLOR_NEAR_WEAK
        osd_img.draw_rectangle(near_info["best_rect"], color=near_color, thickness=3)
        osd_img.draw_cross(
            near_info["best_x"], near_info["best_aim_y"],
            color=near_color, size=11
        )

    draw_text(osd_img, 4, 4, "{} TEAM:{}".format(VERSION, TEAM_COLOR), COLOR_TEXT)
    draw_text(osd_img, 4, 18, "PRIMARY:{}".format(primary_source), COLOR_PRIMARY)
    draw_text(
        osd_img,
        4,
        32,
        "RB:{} BB:{} SAFE:{} AUX:{} CV_OK:{}".format(
            len([item for item in yolo_normal_balls if item["label"] == "redball"]),
            len([item for item in yolo_normal_balls if item["label"] == "blueball"]),
            len(yolo_safes), len(yolo_assists), len(accepted_cv)
        ),
        COLOR_TEXT,
    )
    draw_text(
        osd_img,
        4,
        46,
        "YOLO:{}ms CONV:{}ms CV:{}ms FUSE:{}ms".format(
            timing[0], timing[1], timing[2], timing[3]
        ),
        COLOR_TEXT,
    )
    draw_text(
        osd_img,
        4,
        60,
        "RAW K:{} Y:{} REJ:{} DROP B:{} S:{}".format(
            raw_counts.get("CORE", 0), raw_counts.get("DANGER", 0),
            sum(reject_counts.values()), len(dropped_ball), len(dropped_safe)
        ),
        COLOR_TEXT,
    )
    draw_text(
        osd_img,
        4,
        74,
        "LOCK:{} MODE:{} ARM:{} PEND:{}/{} CAL:{}/{}".format(
            lock_debug.get("locked_type", "NONE"),
            lock_debug.get("mode", "WAIT"),
            1 if lock_debug.get("near_armed") else 0,
            lock_debug.get("pending_count", 0),
            AUTO_LOCK_CONFIRM_FRAMES,
            lock_debug.get("baseline_count", 0),
            EMPTY_BASELINE_CALIBRATION_FRAMES,
        ),
        COLOR_LOCKED,
    )
    if near_info is not None:
        draw_text(
            osd_img,
            4,
            88,
            "NEAR RAW:{} BASE:{} EX:{} A:{:.0f} DX:{} OK:{} C:{}/{}".format(
                near_info.get("raw_pixels", 0),
                lock_debug.get("baseline_pixels", 0),
                lock_debug.get("excess_pixels", 0),
                near_info.get("best_area", 0.0),
                near_info.get("best_dx", 9999),
                1 if lock_debug.get("near_ok") else 0,
                lock_debug.get("near_candidate_count", 0),
                NEAR_CONFIRM_FRAMES,
            ),
            COLOR_NEAR_OK if lock_debug.get("near_ok") else COLOR_NEAR_WEAK,
        )
        draw_text(
            osd_img,
            4,
            102,
            "EVID_AGE:{}ms RELEASE:{}ms".format(
                lock_debug.get("evidence_age_ms", 0),
                lock_debug.get("release_age_ms", 0),
            ),
            COLOR_TEXT,
        )


sensor = None
yolo = None
osd_img = None

try:
    os.exitpoint(os.EXITPOINT_ENABLE)
    print("===== {} =====".format(VERSION))
    print("NO UART / NO MOTOR / NO GRIPPER CONTROL")
    print("objects: YOLO red/blue balls + CV2 black cube/yellow pyramid")
    print("channels: CH0=display CH1=CV-RGB888/HWC CH2=YOLO-RGBP888/CHW")
    print("safe filtering: YOLO normal balls -> YOLO safe zones -> CV2 supplement")
    print("ground valid region: DISABLED")
    print("near-gripper test: AUTO LOCK + RAW HSV + OCCLUSION LATCH")
    print("keep ALL test targets outside NEAR_GRIP_TEST_ROI until EMPTY_BASELINES appears")
    print("then move one target slowly into claw")
    print("TEAM_COLOR={}".format(TEAM_COLOR))
    check_cv2_apis()

    compiled_ranges = compile_hsv_ranges()
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    sensor = Sensor(id=SENSOR_ID)
    sensor.reset()

    sensor.set_framesize(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, chn=DISPLAY_CHN)
    sensor.set_pixformat(Sensor.YUV420SP, chn=DISPLAY_CHN)

    # 官方 Sensor API 支持 CH1 直接输出 RGB888；to_numpy_ref() 为 320x240x3 HWC。
    sensor.set_framesize(width=CV_WIDTH, height=CV_HEIGHT, chn=CV_CHN)
    sensor.set_pixformat(Sensor.RGB888, chn=CV_CHN)

    sensor.set_framesize(width=YOLO_WIDTH, height=YOLO_HEIGHT, chn=YOLO_CHN)
    sensor.set_pixformat(Sensor.RGBP888, chn=YOLO_CHN)

    bind_info = sensor.bind_info(x=0, y=0, chn=DISPLAY_CHN)
    Display.bind_layer(**bind_info, layer=Display.LAYER_VIDEO1)
    Display.init(
        Display.ST7701,
        width=DISPLAY_WIDTH,
        height=DISPLAY_HEIGHT,
        to_ide=True,
        osd_num=1,
    )
    MediaManager.init()
    sensor.run()

    osd_img = image.Image(DISPLAY_WIDTH, DISPLAY_HEIGHT, image.ARGB8888)

    yolo = YOLOv5(
        task_type="detect",
        mode="video",
        kmodel_path=KMODEL_PATH,
        labels=LABELS,
        rgb888p_size=[YOLO_WIDTH, YOLO_HEIGHT],
        model_input_size=MODEL_INPUT_SIZE,
        display_size=[DISPLAY_WIDTH, DISPLAY_HEIGHT],
        conf_thresh=CONFIDENCE_THRESHOLD,
        nms_thresh=NMS_THRESHOLD,
        max_boxes_num=MAX_BOXES_NUM,
        debug_mode=0,
    )
    yolo.config_preprocess()

    fps_clock = time.clock()
    frame_count = 0
    last_cv_candidates = []
    last_rejected_cv = []
    last_raw_counts = {"CORE": 0, "DANGER": 0}
    last_reject_counts = {}
    last_convert_ms = 0
    last_cv_ms = 0
    first_cv_shape_printed = False

    # 自动锁定测试状态。仅用于屏幕/日志诊断，不发送任何控制命令。
    pending_lock_type = None
    pending_lock_count = 0
    pending_lock_x = None
    pending_lock_y = None
    locked_type = None
    locked_last_x = None
    locked_last_y = None
    locked_last_strict_ms = 0
    locked_last_evidence_ms = 0
    near_baseline_pixels = None
    near_armed = False
    near_candidate_count = 0
    lock_mode = "WAIT"
    previous_lock_mode = "WAIT"
    empty_baseline_sums = {"CORE": 0, "DANGER": 0}
    empty_baseline_pixels = {"CORE": None, "DANGER": None}
    empty_baseline_count = 0

    while True:
        os.exitpoint()
        fps_clock.tick()
        frame_count += 1

        # 连续抓取两个处理通道，尽量缩小运动时的跨通道时间差。
        yolo_img = sensor.snapshot(chn=YOLO_CHN)
        cv_img = sensor.snapshot(chn=CV_CHN)
        yolo_input = yolo_img.to_numpy_ref()

        yolo_start = time.ticks_ms()
        raw_result = yolo.run(yolo_input)
        detections = parse_yolo_results(raw_result)
        yolo_ms = time.ticks_diff(time.ticks_ms(), yolo_start)

        current_hsv_img = None
        if frame_count % CV_PROCESS_EVERY_N_FRAMES == 0:
            convert_start = time.ticks_ms()
            small_rgb = cv_img.to_numpy_ref()
            last_convert_ms = time.ticks_diff(time.ticks_ms(), convert_start)

            if not first_cv_shape_printed:
                print("CV RGB888 HWC shape:", small_rgb.shape)
                if len(small_rgb.shape) != 3 or small_rgb.shape[2] != 3:
                    raise RuntimeError(
                        "CH1 RGB888 expected HWC (240,320,3), got {}".format(
                            small_rgb.shape
                        )
                    )
                first_cv_shape_printed = True

            cv_start = time.ticks_ms()
            # Sensor RGB888 是 RGB 排列，不能使用 COLOR_BGR2HSV。
            current_hsv_img = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
            (
                last_cv_candidates,
                last_rejected_cv,
                last_raw_counts,
                last_reject_counts,
            ) = collect_cv_materials(
                current_hsv_img, compiled_ranges, open_kernel, close_kernel
            )
            last_cv_ms = time.ticks_diff(time.ticks_ms(), cv_start)

            if empty_baseline_count < EMPTY_BASELINE_CALIBRATION_FRAMES:
                for baseline_type in CV_MATERIAL_ORDER:
                    empty_baseline_sums[baseline_type] += measure_near_raw_pixels(
                        current_hsv_img, baseline_type, compiled_ranges
                    )
                empty_baseline_count += 1
                if empty_baseline_count == EMPTY_BASELINE_CALIBRATION_FRAMES:
                    for baseline_type in CV_MATERIAL_ORDER:
                        empty_baseline_pixels[baseline_type] = int(
                            empty_baseline_sums[baseline_type] /
                            EMPTY_BASELINE_CALIBRATION_FRAMES
                        )
                    print(
                        ">>> EMPTY_BASELINES READY CORE={} DANGER={}".format(
                            empty_baseline_pixels["CORE"],
                            empty_baseline_pixels["DANGER"]
                        )
                    )
            del small_rgb

        fusion_start = time.ticks_ms()
        yolo_normal_balls = []
        yolo_safes = []
        yolo_assists = []
        for det in detections:
            if det["label"] in YOLO_NORMAL_BALL_LABELS:
                yolo_normal_balls.append(det)
            elif det["label"] in YOLO_SAFE_LABELS:
                yolo_safes.append(det)
            elif det["label"] in YOLO_ASSIST_LABELS:
                yolo_assists.append(det)

        accepted_cv, dropped_ball, dropped_safe = fuse_cv_candidates(
            last_cv_candidates, yolo_normal_balls, yolo_safes, yolo_assists
        )
        primary_source, primary_target = choose_primary_target(
            yolo_normal_balls, accepted_cv
        )

        now_ms = time.ticks_ms()
        strict_locked_target = None
        near_info = None
        near_ok = False
        excess_pixels = 0
        evidence_age_ms = 0
        release_age_ms = 0

        # 没有锁定时，只允许严格 CV 候选连续出现后建立锁定。
        if locked_type is None:
            cv_primary_type = None
            if empty_baseline_count >= EMPTY_BASELINE_CALIBRATION_FRAMES:
                if primary_source == "CV2_CORE":
                    cv_primary_type = "CORE"
                elif primary_source == "CV2_DANGER":
                    cv_primary_type = "DANGER"

            if cv_primary_type is not None and primary_target is not None:
                primary_x = primary_target.get("aim_x", primary_target["cx"])
                primary_y = primary_target.get("aim_y", primary_target["cy"])
                same_pending = pending_lock_type == cv_primary_type
                if same_pending and pending_lock_x is not None and pending_lock_y is not None:
                    dx = primary_x - pending_lock_x
                    dy = primary_y - pending_lock_y
                    same_pending = (
                        dx * dx + dy * dy <=
                        STRICT_MATCH_MAX_DISTANCE * STRICT_MATCH_MAX_DISTANCE
                    )

                if same_pending:
                    pending_lock_count += 1
                else:
                    pending_lock_type = cv_primary_type
                    pending_lock_count = 1
                pending_lock_x = primary_x
                pending_lock_y = primary_y

                if pending_lock_count >= AUTO_LOCK_CONFIRM_FRAMES:
                    locked_type = cv_primary_type
                    locked_last_x = primary_x
                    locked_last_y = primary_y
                    locked_last_strict_ms = now_ms
                    locked_last_evidence_ms = 0
                    near_baseline_pixels = empty_baseline_pixels[cv_primary_type]
                    near_armed = False
                    near_candidate_count = 0
                    lock_mode = "STRICT"
                    print(
                        ">>> AUTO_LOCK type={} aim=({}, {}) baseline={} after {} strict frames".format(
                            locked_type, locked_last_x, locked_last_y,
                            near_baseline_pixels,
                            pending_lock_count
                        )
                    )
                    pending_lock_type = None
                    pending_lock_count = 0
                    pending_lock_x = None
                    pending_lock_y = None
            else:
                pending_lock_type = None
                pending_lock_count = 0
                pending_lock_x = None
                pending_lock_y = None
                lock_mode = "WAIT"
        else:
            pending_lock_type = None
            pending_lock_count = 0

        if locked_type is not None:
            strict_locked_target = select_locked_strict_target(
                accepted_cv, locked_type, locked_last_x, locked_last_y,
                near_armed
            )
            near_config = NEAR_TRACK_CONFIG[locked_type]

            if strict_locked_target is not None:
                locked_last_x = strict_locked_target.get(
                    "aim_x", strict_locked_target["cx"]
                )
                locked_last_y = strict_locked_target.get(
                    "aim_y", strict_locked_target["cy"]
                )
                locked_last_strict_ms = now_ms
                if locked_last_y >= near_config["enter_aim_y"]:
                    if not near_armed:
                        print(
                            ">>> NEAR_ARMED type={} aim_y={} threshold={}".format(
                                locked_type, locked_last_y,
                                near_config["enter_aim_y"]
                            )
                        )
                    near_armed = True
                    locked_last_evidence_ms = now_ms

            if current_hsv_img is not None:
                near_info = collect_near_grip_evidence(
                    current_hsv_img, locked_type, compiled_ranges,
                    open_kernel, close_kernel, locked_last_x
                )
                if near_baseline_pixels is None:
                    near_baseline_pixels = near_info["raw_pixels"]
                    print(
                        ">>> NEAR_BASELINE type={} raw_pixels={}".format(
                            locked_type, near_baseline_pixels
                        )
                    )

                excess_pixels = near_info["raw_pixels"] - near_baseline_pixels
                if excess_pixels < 0:
                    excess_pixels = 0
                near_candidate_ok = (
                    near_info["best_rect"] is not None and
                    near_info["best_area"] >= near_config["min_contour_area"] and
                    near_info["best_dx"] <= near_config["max_x_delta"] and
                    excess_pixels >= near_config["min_excess_pixels"]
                )
                if near_candidate_ok:
                    near_candidate_count += 1
                    if near_candidate_count > NEAR_CONFIRM_FRAMES:
                        near_candidate_count = NEAR_CONFIRM_FRAMES
                else:
                    near_candidate_count = 0
                near_ok = near_candidate_count >= NEAR_CONFIRM_FRAMES

            if strict_locked_target is not None:
                lock_mode = "STRICT"
            elif near_armed and near_ok:
                lock_mode = "NEAR"
                locked_last_x = near_info["best_x"]
                locked_last_y = near_info["best_aim_y"]
                locked_last_evidence_ms = now_ms
            elif near_armed and locked_last_evidence_ms != 0:
                evidence_age_ms = time.ticks_diff(now_ms, locked_last_evidence_ms)
                if evidence_age_ms <= NEAR_LATCH_MS:
                    lock_mode = "LATCH"
                else:
                    lock_mode = "LOST"
            else:
                lock_mode = "LOST"

            if locked_last_evidence_ms != 0:
                evidence_age_ms = time.ticks_diff(now_ms, locked_last_evidence_ms)
            last_seen_ms = locked_last_strict_ms
            if locked_last_evidence_ms != 0 and time.ticks_diff(
                    locked_last_evidence_ms, last_seen_ms) > 0:
                last_seen_ms = locked_last_evidence_ms
            release_age_ms = time.ticks_diff(now_ms, last_seen_ms)

            if release_age_ms > LOCK_RELEASE_MS:
                print(
                    ">>> AUTO_UNLOCK type={} after {}ms without strict/near evidence".format(
                        locked_type, release_age_ms
                    )
                )
                locked_type = None
                locked_last_x = None
                locked_last_y = None
                locked_last_strict_ms = 0
                locked_last_evidence_ms = 0
                near_baseline_pixels = None
                near_armed = False
                near_candidate_count = 0
                strict_locked_target = None
                near_info = None
                near_ok = False
                excess_pixels = 0
                evidence_age_ms = 0
                release_age_ms = 0
                lock_mode = "WAIT"

        if lock_mode != previous_lock_mode:
            print(
                ">>> LOCK_MODE {} -> {} type={}".format(
                    previous_lock_mode, lock_mode,
                    locked_type if locked_type is not None else "NONE"
                )
            )
            previous_lock_mode = lock_mode

        lock_debug = {
            "locked_type": locked_type if locked_type is not None else "NONE",
            "mode": lock_mode,
            "near_armed": near_armed,
            "pending_count": pending_lock_count,
            "strict_target": strict_locked_target,
            "near_info": near_info,
            "near_ok": near_ok,
            "near_candidate_count": near_candidate_count,
            "baseline_pixels": (
                near_baseline_pixels if near_baseline_pixels is not None else 0
            ),
            "excess_pixels": excess_pixels,
            "evidence_age_ms": evidence_age_ms,
            "release_age_ms": release_age_ms,
            "baseline_count": empty_baseline_count,
        }
        fusion_ms = time.ticks_diff(time.ticks_ms(), fusion_start)

        draw_results(
            osd_img,
            detections,
            yolo_normal_balls,
            yolo_safes,
            yolo_assists,
            accepted_cv,
            last_rejected_cv,
            dropped_ball,
            dropped_safe,
            primary_source,
            primary_target,
            (yolo_ms, last_convert_ms, last_cv_ms, fusion_ms),
            last_raw_counts,
            last_reject_counts,
            lock_debug,
        )
        Display.show_image(osd_img, 0, 0, Display.LAYER_OSD0)

        if frame_count % DEBUG_PRINT_EVERY_N_FRAMES == 0:
            print(
                "frame={} fps={:.2f} yolo={}ms convert={}ms cv={}ms fuse={}ms "
                "red={} blue={} safe={} aux={} cv_raw_k={} cv_raw_y={} "
                "cv_ok={} rejected={} drop_ball={} drop_safe={} primary={} mem={}".format(
                    frame_count,
                    fps_clock.fps(),
                    yolo_ms,
                    last_convert_ms,
                    last_cv_ms,
                    fusion_ms,
                    len([item for item in yolo_normal_balls if item["label"] == "redball"]),
                    len([item for item in yolo_normal_balls if item["label"] == "blueball"]),
                    len(yolo_safes),
                    len(yolo_assists),
                    last_raw_counts.get("CORE", 0),
                    last_raw_counts.get("DANGER", 0),
                    len(accepted_cv),
                    sum(last_reject_counts.values()),
                    len(dropped_ball),
                    len(dropped_safe),
                    primary_source,
                    free_memory_text(),
                )
            )
            print("reject_counts:", last_reject_counts)
            for target in accepted_cv:
                print(
                    "  CV_OK type={} center=({}, {}) aim=({}, {}) wh=({}, {}) "
                    "area320={:.0f} aspect={:.2f} fill={:.2f} contrast={:.1f} "
                    "v_in={:.1f} v_out={:.1f} vertices={} yolo_support={}".format(
                        target["type"], target["cx"], target["cy"],
                        target["aim_x"], target["aim_y"],
                        target["w"], target["h"],
                        target["contour_area_320"], target["aspect"],
                        target["fill"], target["contrast"],
                        target["inner_v"], target["outer_v"],
                        target["vertices"], target["yolo_support"],
                    )
                )
            if locked_type is not None:
                if near_info is None:
                    print(
                        "  LOCK type={} mode={} armed={} near=NO_NEW_CV_FRAME "
                        "evidence_age={}ms release_age={}ms".format(
                            locked_type, lock_mode, near_armed,
                            evidence_age_ms, release_age_ms
                        )
                    )
                else:
                    print(
                        "  LOCK type={} mode={} armed={} strict={} near_ok={} "
                        "raw={} baseline={} excess={} contours={} best_area={:.0f} "
                        "best_dx={} confirm={}/{} evidence_age={}ms release_age={}ms".format(
                            locked_type, lock_mode, near_armed,
                            strict_locked_target is not None, near_ok,
                            near_info["raw_pixels"],
                            near_baseline_pixels,
                            excess_pixels,
                            near_info["contours"],
                            near_info["best_area"],
                            near_info["best_dx"],
                            near_candidate_count,
                            NEAR_CONFIRM_FRAMES,
                            evidence_age_ms,
                            release_age_ms,
                        )
                    )
            gc.collect()

        if current_hsv_img is not None:
            del current_hsv_img
        del yolo_input
        del yolo_img
        del cv_img
        time.sleep_ms(5)

except KeyboardInterrupt as error:
    print("user stopped:", error)
except BaseException as error:
    if str(error) == "IDE interrupt":
        print("user stopped by IDE")
    else:
        print("final material test FAILED:", error)
finally:
    try:
        if sensor is not None:
            sensor.stop()
    except Exception:
        pass
    try:
        if yolo is not None:
            yolo.deinit()
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
    print("final material test finished")
