# ============================================================
# K230 YOLO + LAB 双识别通路独立测试
# YOLO 球体优先；LAB 候选依次排除 YOLO 球体和安全区后再作为补充目标。
# 本测试不含地面有效区、状态机、串口和底盘控制。
# ============================================================

import gc
import os
import time
import image

from media.sensor import *
from media.display import *
from media.media import MediaManager
from libs.YOLO import YOLOv5


VERSION = "K230 YOLO+LAB dual path test v1.0"

# ===================== camera / display =====================
SENSOR_ID = 1
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 480
YOLO_WIDTH = 800
YOLO_HEIGHT = 480
LAB_WIDTH = 320
LAB_HEIGHT = 240

# 三路互不覆盖：
# CH0: LCD/IDE 背景画面；CH1: LAB RGB565；CH2: YOLO RGBP888。
DISPLAY_CHN = CAM_CHN_ID_0
LAB_CHN = CAM_CHN_ID_1
YOLO_CHN = CAM_CHN_ID_2

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

# 当前主程序的安全区类别是 redboard / blueboard。
# redsection / bluesection 等场地标志不会在本测试里冒充“安全区”。
YOLO_BALL_LABELS = ("redball", "blueball", "blackball", "yellowball")
YOLO_SAFE_LABELS = ("redboard", "blueboard")

# 只影响屏幕顶部的“最终优先目标”判断，不影响空间去重。
# 即使检测到对方普通球，它仍会用于压掉重叠 LAB 框，避免被 LAB 重新认成合法目标。
TEAM_COLOR = "BLUE"  # 现场改成 "RED" 或 "BLUE"

# ===================== LAB thresholds =====================
# CanMV IDE LAB 阈值编辑器格式：
# (L Min, L Max, A Min, A Max, B Min, B Max)
RED_LAB_THRESHOLD = (18, 54, 30, 60, -6, 51)
BLUE_LAB_THRESHOLD = (19, 50, 0, 59, -82, -33)
YELLOW_LAB_THRESHOLD = (38, 83, -29, -5, 34, 122)

# 黑色与阴影、衣物、暗背景难以区分，继续只交给 YOLO。
COLOR_CONFIGS = [
    ("RED", RED_LAB_THRESHOLD, (255, 0, 0)),
    ("BLUE", BLUE_LAB_THRESHOLD, (0, 128, 255)),
    ("YELLOW", YELLOW_LAB_THRESHOLD, (255, 255, 0)),
]

# find_blobs 初筛参数，单位均基于 320x240 LAB 图像。
PIXELS_THRESHOLD = 40
AREA_THRESHOLD = 50
MIN_BOX_AREA = 50
MAX_BOX_AREA = 30000
MIN_BOX_WIDTH = 5
MIN_BOX_HEIGHT = 5
MIN_DENSITY = 0.25
MAX_ASPECT_RATIO = 4.0

# 跨通道图像存在很小的时间差和框大小差异，因此不用单一 IoU 判断。
# 满足“LAB 中心落入外扩 YOLO 框”或“LAB 框至少 20% 被覆盖”即视为重复。
YOLO_OVERLAP_MARGIN = 12       # 800x480 坐标中的像素
LAB_COVERED_RATIO_MIN = 0.20

# 可视化颜色。
COLOR_YOLO_BALL = (0, 255, 0)
COLOR_YOLO_SAFE = (255, 0, 255)
COLOR_LAB_ACCEPT = (0, 255, 255)
COLOR_LAB_DROP_BALL = (120, 120, 120)
COLOR_LAB_DROP_SAFE = (255, 128, 0)
COLOR_TEXT = (255, 255, 255)


def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def draw_text(img, x, y, text, color=COLOR_TEXT):
    x = clamp(int(x), 0, DISPLAY_WIDTH - 8)
    y = clamp(int(y), 0, DISPLAY_HEIGHT - 12)
    try:
        img.draw_string(x, y, text, color=color, scale=1)
    except Exception:
        pass


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

    # 当前 K230 aidemo.yolov5_det_postprocess 的标准输出：
    # (boxes, class_ids, scores)，其中 box 为 [x, y, width, height]。
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

    # 保留对常见列表 / 字典 / 对象格式的兼容。
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
    return aspect_ratio <= MAX_ASPECT_RATIO


def decode_blob_color(blob_code):
    matched = []
    for index in range(len(COLOR_CONFIGS)):
        if blob_code & (1 << index):
            matched.append(index)
    if len(matched) == 1:
        config = COLOR_CONFIGS[matched[0]]
        return config[0], config[2]
    if len(matched) > 1:
        return "MIXED", (255, 255, 255)
    return "UNKNOWN", (255, 255, 255)


def map_lab_rect_to_yolo(blob):
    x1 = int(blob.x() * YOLO_WIDTH / LAB_WIDTH)
    y1 = int(blob.y() * YOLO_HEIGHT / LAB_HEIGHT)
    width = max(1, int(blob.w() * YOLO_WIDTH / LAB_WIDTH))
    height = max(1, int(blob.h() * YOLO_HEIGHT / LAB_HEIGHT))
    x2 = clamp(x1 + width, 0, YOLO_WIDTH)
    y2 = clamp(y1 + height, 0, YOLO_HEIGHT)
    x1 = clamp(x1, 0, YOLO_WIDTH - 1)
    y1 = clamp(y1, 0, YOLO_HEIGHT - 1)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return (x1, y1, width, height)


def collect_lab_candidates(lab_img, thresholds):
    blobs = lab_img.find_blobs(
        thresholds,
        pixels_threshold=PIXELS_THRESHOLD,
        area_threshold=AREA_THRESHOLD,
        merge=False,
        margin=0,
    )

    candidates = []
    for blob in blobs:
        if not is_valid_blob(blob):
            continue
        color_name, draw_color = decode_blob_color(blob.code())
        if color_name == "UNKNOWN" or color_name == "MIXED":
            continue

        rect = map_lab_rect_to_yolo(blob)
        candidates.append({
            "source": "LAB",
            "color": color_name,
            "draw_color": draw_color,
            "rect": rect,
            "x1": rect[0],
            "y1": rect[1],
            "x2": rect[0] + rect[2],
            "y2": rect[1] + rect[3],
            "w": rect[2],
            "h": rect[3],
            "cx": rect[0] + rect[2] // 2,
            "cy": rect[1] + rect[3] // 2,
            "area": rect[2] * rect[3],
            "pixels_320": blob.pixels(),
            "density": blob.density(),
        })
    return blobs, candidates


def lab_overlaps_yolo(lab_target, yolo_target):
    margin = YOLO_OVERLAP_MARGIN
    yx1 = clamp(yolo_target["x1"] - margin, 0, YOLO_WIDTH)
    yy1 = clamp(yolo_target["y1"] - margin, 0, YOLO_HEIGHT)
    yx2 = clamp(yolo_target["x2"] + margin, 0, YOLO_WIDTH)
    yy2 = clamp(yolo_target["y2"] + margin, 0, YOLO_HEIGHT)

    # 色块中心位于 YOLO 目标中，是不同检测框尺度下最可靠的重复信号。
    if yx1 <= lab_target["cx"] <= yx2 and yy1 <= lab_target["cy"] <= yy2:
        return True

    ix1 = max(lab_target["x1"], yx1)
    iy1 = max(lab_target["y1"], yy1)
    ix2 = min(lab_target["x2"], yx2)
    iy2 = min(lab_target["y2"], yy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return False

    intersection = (ix2 - ix1) * (iy2 - iy1)
    return intersection / lab_target["area"] >= LAB_COVERED_RATIO_MIN


def fuse_candidates(lab_candidates, yolo_balls, yolo_safes):
    """严格按 YOLO 球体 -> YOLO 安全区 -> LAB 补充目标的顺序过滤。"""
    accepted = []
    dropped_ball = []
    dropped_safe = []

    for lab_target in lab_candidates:
        matched = None
        for yolo_target in yolo_balls:
            if lab_overlaps_yolo(lab_target, yolo_target):
                matched = yolo_target
                break
        if matched is not None:
            lab_target["drop_reason"] = "BALL:{}".format(matched["label"])
            dropped_ball.append(lab_target)
            continue

        for yolo_target in yolo_safes:
            if lab_overlaps_yolo(lab_target, yolo_target):
                matched = yolo_target
                break
        if matched is not None:
            lab_target["drop_reason"] = "SAFE:{}".format(matched["label"])
            dropped_safe.append(lab_target)
            continue

        accepted.append(lab_target)

    return accepted, dropped_ball, dropped_safe


def is_legal_yolo_ball(target):
    own_ball = "redball" if TEAM_COLOR == "RED" else "blueball"
    return target["label"] == own_ball or target["label"] in ("blackball", "yellowball")


def is_legal_lab_target(target):
    # LAB 黑色关闭；普通物块只允许本方颜色，黄色作为公共危险目标保留。
    return target["color"] == TEAM_COLOR or target["color"] == "YELLOW"


def choose_primary_target(yolo_balls, accepted_lab):
    legal_yolo = []
    for target in yolo_balls:
        if is_legal_yolo_ball(target):
            legal_yolo.append(target)
    if legal_yolo:
        return "YOLO", max(legal_yolo, key=lambda item: item["area"])

    legal_lab = []
    for target in accepted_lab:
        if is_legal_lab_target(target):
            legal_lab.append(target)
    if legal_lab:
        return "LAB", max(legal_lab, key=lambda item: item["area"])
    return "NONE", None


def draw_result(osd_img, detections, yolo_balls, yolo_safes,
                accepted_lab, dropped_ball, dropped_safe,
                primary_source, primary_target, timing):
    osd_img.clear()

    # 只突出显示用于融合的 YOLO 球体和安全区；其他模型类别用细灰框显示。
    for det in detections:
        if det["label"] in YOLO_BALL_LABELS:
            color = COLOR_YOLO_BALL
            prefix = "Y-BALL"
            thickness = 2
        elif det["label"] in YOLO_SAFE_LABELS:
            color = COLOR_YOLO_SAFE
            prefix = "Y-SAFE"
            thickness = 3
        else:
            color = (90, 90, 90)
            prefix = "Y-OTHER"
            thickness = 1
        osd_img.draw_rectangle(det["rect"], color=color, thickness=thickness)
        draw_text(
            osd_img,
            det["x1"],
            det["y1"] - 12,
            "{} {} {:.2f}".format(prefix, det["label"], det["score"]),
            color,
        )

    for target in dropped_ball:
        osd_img.draw_rectangle(target["rect"], color=COLOR_LAB_DROP_BALL, thickness=1)
        draw_text(osd_img, target["x1"], target["y1"] + target["h"],
                  "DROP-{}".format(target["drop_reason"]), COLOR_LAB_DROP_BALL)

    for target in dropped_safe:
        osd_img.draw_rectangle(target["rect"], color=COLOR_LAB_DROP_SAFE, thickness=1)
        draw_text(osd_img, target["x1"], target["y1"] + target["h"],
                  "DROP-{}".format(target["drop_reason"]), COLOR_LAB_DROP_SAFE)

    for target in accepted_lab:
        legal = is_legal_lab_target(target)
        color = COLOR_LAB_ACCEPT if legal else (180, 100, 180)
        text_prefix = "LAB-NEW" if legal else "LAB-OPP"
        osd_img.draw_rectangle(target["rect"], color=color, thickness=3 if legal else 1)
        osd_img.draw_cross(target["cx"], target["cy"], color=color, size=8)
        draw_text(osd_img, target["x1"], target["y1"] - 12,
                  "{} {}".format(text_prefix, target["color"]), color)

    # 最终目标用白色粗框强调；这只是测试输出，不发送给下位机。
    if primary_target is not None:
        osd_img.draw_rectangle(primary_target["rect"], color=(255, 255, 255), thickness=4)
        osd_img.draw_cross(primary_target["cx"], primary_target["cy"],
                           color=(255, 255, 255), size=12)

    draw_text(osd_img, 4, 4, "{} TEAM:{}".format(VERSION, TEAM_COLOR), COLOR_TEXT)
    draw_text(osd_img, 4, 18, "PRIMARY:{}".format(primary_source), COLOR_TEXT)
    draw_text(
        osd_img,
        4,
        32,
        "YBALL:{} YSAFE:{} LAB_OK:{} DROP_B:{} DROP_S:{}".format(
            len(yolo_balls), len(yolo_safes), len(accepted_lab),
            len(dropped_ball), len(dropped_safe)
        ),
        COLOR_TEXT,
    )
    draw_text(
        osd_img,
        4,
        46,
        "YOLO:{}ms LAB:{}ms FUSE:{}ms".format(
            timing[0], timing[1], timing[2]
        ),
        COLOR_TEXT,
    )


sensor = None
yolo = None
osd_img = None

try:
    os.exitpoint(os.EXITPOINT_ENABLE)
    print("===== {} =====".format(VERSION))
    print("channels: CH0=display CH1=LAB-RGB565 CH2=YOLO-RGBP888")
    print("TEAM_COLOR={}".format(TEAM_COLOR))
    print("LAB filter order: all YOLO balls -> all YOLO safe zones -> LAB supplement")
    print("ground valid region: DISABLED")

    sensor = Sensor(id=SENSOR_ID)
    sensor.reset()

    sensor.set_framesize(width=DISPLAY_WIDTH, height=DISPLAY_HEIGHT, chn=DISPLAY_CHN)
    sensor.set_pixformat(Sensor.YUV420SP, chn=DISPLAY_CHN)

    sensor.set_framesize(width=YOLO_WIDTH, height=YOLO_HEIGHT, chn=YOLO_CHN)
    sensor.set_pixformat(Sensor.RGBP888, chn=YOLO_CHN)

    sensor.set_framesize(width=LAB_WIDTH, height=LAB_HEIGHT, chn=LAB_CHN)
    sensor.set_pixformat(Sensor.RGB565, chn=LAB_CHN)

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

    thresholds = []
    for config in COLOR_CONFIGS:
        thresholds.append(config[1])

    fps_clock = time.clock()
    frame_count = 0

    while True:
        os.exitpoint()
        fps_clock.tick()
        frame_count += 1

        # 两个通道先连续取帧，再分别处理，尽量缩小移动中跨通道的时间偏差。
        yolo_img = sensor.snapshot(chn=YOLO_CHN)
        lab_img = sensor.snapshot(chn=LAB_CHN)

        yolo_start = time.ticks_ms()
        yolo_input = yolo_img.to_numpy_ref()
        raw_result = yolo.run(yolo_input)
        detections = parse_yolo_results(raw_result)
        yolo_ms = time.ticks_diff(time.ticks_ms(), yolo_start)

        lab_start = time.ticks_ms()
        raw_blobs, lab_candidates = collect_lab_candidates(lab_img, thresholds)
        lab_ms = time.ticks_diff(time.ticks_ms(), lab_start)

        fusion_start = time.ticks_ms()
        yolo_balls = []
        yolo_safes = []
        for det in detections:
            if det["label"] in YOLO_BALL_LABELS:
                yolo_balls.append(det)
            elif det["label"] in YOLO_SAFE_LABELS:
                yolo_safes.append(det)

        accepted_lab, dropped_ball, dropped_safe = fuse_candidates(
            lab_candidates, yolo_balls, yolo_safes
        )
        primary_source, primary_target = choose_primary_target(yolo_balls, accepted_lab)
        fusion_ms = time.ticks_diff(time.ticks_ms(), fusion_start)

        draw_result(
            osd_img,
            detections,
            yolo_balls,
            yolo_safes,
            accepted_lab,
            dropped_ball,
            dropped_safe,
            primary_source,
            primary_target,
            (yolo_ms, lab_ms, fusion_ms),
        )
        Display.show_image(osd_img, 0, 0, Display.LAYER_OSD0)

        if frame_count % 30 == 0:
            print(
                "frame={} fps={:.2f} yolo={}ms lab={}ms fuse={}ms "
                "det={} yball={} ysafe={} lab_raw={} lab_valid={} "
                "drop_ball={} drop_safe={} lab_ok={} primary={}".format(
                    frame_count,
                    fps_clock.fps(),
                    yolo_ms,
                    lab_ms,
                    fusion_ms,
                    len(detections),
                    len(yolo_balls),
                    len(yolo_safes),
                    len(raw_blobs),
                    len(lab_candidates),
                    len(dropped_ball),
                    len(dropped_safe),
                    len(accepted_lab),
                    primary_source,
                )
            )
            for target in accepted_lab:
                print(
                    "  LAB_OK {} center=({}, {}) wh=({}, {}) density={:.2f} legal={}".format(
                        target["color"], target["cx"], target["cy"],
                        target["w"], target["h"], target["density"],
                        is_legal_lab_target(target),
                    )
                )
            gc.collect()

        del yolo_input
        del yolo_img
        del lab_img
        time.sleep_ms(5)

except KeyboardInterrupt as error:
    print("user stopped:", error)
except BaseException as error:
    if str(error) == "IDE interrupt":
        print("user stopped by IDE")
    else:
        print("dual path test FAILED:", error)
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
    print("dual path test finished")
