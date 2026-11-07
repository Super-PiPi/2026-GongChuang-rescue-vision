# ============================================================
# 工创赛救援小车 K230 视觉上位机程序
# 主要功能：
# 1. 识别球类目标：己方普通球、对方普通球、危险球、核心球
# 2. 按优先级选择目标，并通过串口发送运动/夹爪指令给 STM32 下位机
# 3. 抓取目标后识别安全区，运输到安全区并执行卸载流程
# 4. 状态机流程：SEARCH -> TRACK -> GRAB -> TRANSPORT -> UNLOAD
# 2026.6.25
##加入了对安全区内小球的过滤逻辑，避免抓取安全区内的球。加入了卸载后的冷却时间，使小车在卸载后不会正对着安全区。
#修改了对准自校正的标志位，加入了实时的signdebug显示，方便调试。
# ============================================================

import time
import os
import image
from media.sensor import *
from media.media import MediaManager
from machine import UART
from machine import FPIOA
from libs.PipeLine import PipeLine
from libs.YOLO import YOLOv5
from libs.Utils import *
import ulab.numpy as np
import gc

VERSION = "Vision Full FSM + fixed transport/unload + cooldown + stop align"

# ===================== hardware =====================
# 摄像头与图像尺寸相关配置。
# 使用 PipeLine 获取帧并显示，当前 rgb888p_size 与 YOLO 输入尺寸匹配。
sensor_id = 1
sensor = None
pl = None
yolo = None
SENSOR_WIDTH = 800
SENSOR_HEIGHT = 480
CENTER_X = SENSOR_WIDTH // 2

# ===================== model config =====================
KMODEL_PATH = "/sdcard/model/best_0719.kmodel"
#KMODEL_PATH="/sdcard/model/640_best.kmodel"
MODEL_INPUT_SIZE = [320, 320]
labels = [
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
LABELS = labels

display_mode = "lcd"
display_size = None
rgb888p_size = [800, 480]
confidence_threshold = 0.3 #yolo置信度
nms_threshold = 0.45   #NMS 重叠抑制阈值；影响重复框和目标稳定。
max_boxes_num = 50     # YOLOv5 识别框数量；影响识别速度和误检。

gc.collect()

# ===================== team / thresholds =====================
# 队伍颜色决定抓取顺序与目标优先级。
TEAM_COLOR = "BLUE"  # 比赛前修改队伍颜色： "RED" 或 "BLUE"

# ===================== target weights (on-site config) =====================
# ⚠️ 现场调参区：根据赛场当天公布的规则，在此处修改各项小球的得分权重！
# 权重越高，小车在视野内同时看到多种球时，越优先去抓它。
WEIGHT_NORMAL = 20  # 本方普通球的分值权重（确保它足够大，优先抓自己的球）
WEIGHT_CORE   = 1   # 核心球(黑色)的分值权重
WEIGHT_DANGER = 5   # 危险球(黄色)的分值权重
# 注意：代码已从底层屏蔽敌方普通球，无需在此处配置敌方权重，绝不资敌！

# ===================== target filter parameters =====================
# YOLO 目标球过滤参数，只用于筛掉太小/异常大的检测框，不作为到位判断。
# 需要根据摄像头视角和误检情况现场调参。
MIN_DRAW_AREA = 50
MAX_DRAW_AREA = 80000

# 目标优先级打分参数，偏离画面中心越远惩罚越大。
# 影响 SEARCH/TRACK 选择哪个球，可能需要根据赛场目标密度调参。
CENTER_SCORE_PENALTY = 40

# ===================== track/grab state parameters =====================
# TRACK 阶段球中心 x 允许误差，单位像素。
# 用于 TRACK -> GRAB，需要根据下位机 BALL_XY 横向 PID 稳定性现场调参。
TRACK_X_DEAD_ZONE = 80

# TRACK 阶段进入 GRAB 的球中心 y 目标值，单位像素。
# 必须和下位机 BALL_XY 模式下的 y 目标保持一致，否则上下位机状态判断会错位。
TRACK_GRAB_TARGET_Y = 420

# SEARCH 阶段连续锁定目标帧数，抑制偶发误检。
# 需要根据识别稳定性现场调参。
SEARCH_ENTER_TRACK_FRAMES = 3

# SEARCH 进入 TRACK 的最小检测面积，只作为误检过滤，不作为到位判断。
# 需要根据目标球远近和误检情况现场调参。
SEARCH_LOCK_MIN_AREA = 100

# 【新增】卸载完成后的强制致盲转身时间（毫秒）。防止立刻看到刚卸载的球又去抓。
SEARCH_COOLDOWN_MS = 1500

# TRACK -> GRAB 的 x/y 到位连续确认帧数。
# 需要根据画面抖动和底盘响应速度现场调参。
GRAB_CONFIRM_FRAMES = 3

# 夹爪 ROI，用于 GRAB 阶段视觉确认球是否进入夹取区域。
# 需要根据摄像头安装位置和夹爪位置现场调参。
GRIP_ROI = (320, 390, 170, 85)

# GRAB 阶段 ROI 内球检测过滤面积，只用于误检过滤。
# 需要根据夹爪附近目标大小现场调参。
GRIP_BALL_MIN_AREA = 1000

# GRAB 阶段 ROI 连续确认帧数，抑制偶发误检。
# 需要根据画面稳定性现场调参。
GRIP_BALL_CONFIRM_FRAMES = 3

# 夹爪闭合保持时间，单位 ms。
# 需要根据舵机速度和实际抓取稳定性现场调参。
GRIP_CLOSE_TIME_MS = 1200

# ===================== safe zone filter parameters =====================
# 安全区识别过滤参数。当前逻辑主要利用：
# 1. 安全区内部颜色：蓝队/红队对应的色块
# 2. 紫色边框：用于排除普通蓝/红色区域，把安全区和其他区域区分开
# 这些面积/尺寸只用于检测过滤，不作为 TRANSPORT 到位判断。
# 需要根据安全区实际大小、相机视角和误检情况现场调参。
SAFE_ZONE_DETECT_MIN_AREA = 1200
SAFE_ZONE_MIN_WIDTH = 50
SAFE_ZONE_MIN_HEIGHT = 1

# UNLOAD 固定动作结束后的状态恢复计数，目前保留为卸载状态内部计数。
UNLOAD_CONFIRM_FRAMES = 2

# ===================== transport xy/self-align parameters =====================
# TRANSPORT 阶段安全区中心 x 允许误差，单位像素。
# 用于 TRANSPORT -> SELF_ALIGN，需要根据下位机 SAFE_XY 横向 PID 稳定性现场调参。
TRANSPORT_X_DEAD_ZONE = 150

# TRANSPORT 阶段安全区中心 y 到位目标值，单位像素。
# 必须和下位机 SAFE_XY 模式下的 y 目标保持一致。
TRANSPORT_SAFE_TARGET_Y = 140

# TRANSPORT -> SELF_ALIGN 的 x/y 到位连续确认帧数。
# 需要根据安全区识别抖动和底盘响应速度现场调参。
TRANSPORT_XY_CONFIRM_FRAMES = 3

# SAFE_XY 模式下复用 angle 字段的自矫正标志。
# NONE=普通靠近，START=请求下位机持续执行自身矫正。
TRANSPORT_ALIGN_FLAG_NONE = 0
TRANSPORT_ALIGN_FLAG_START = 1

# TRANSPORT 内部两阶段：x/y 靠近，以及请求下位机自矫正。
TRANSPORT_PHASE_XY_APPROACH = 0
TRANSPORT_PHASE_SELF_ALIGN = 1

# SELF_ALIGN 至少等待时间和最大等待时间，单位 ms。
# 需要根据下位机自矫正动作耗时现场调参；TIMEOUT 防止状态卡死。
ALIGN_WAIT_MIN_MS = 3000
ALIGN_WAIT_TIMEOUT_MS = 5000

# SELF_ALIGN 中画面变化阈值，判断下位机矫正是否产生可见运动。
# 只用于 debug/辅助观察，不强制要求 seen_motion 为 True 才进入 UNLOAD。
ALIGN_MOVE_X_TH = 20
ALIGN_MOVE_Y_TH = 15

# SELF_ALIGN 中画面稳定阈值和稳定帧数。
# 用于 SELF_ALIGN -> UNLOAD，需要根据识别抖动现场调参。
ALIGN_STABLE_X_TH = 8
ALIGN_STABLE_Y_TH = 8
ALIGN_STABLE_CONFIRM_FRAMES = 5

if TEAM_COLOR == "BLUE":
    SAFE_ZONE_DRAW_COLOR = (0, 0, 255)
else:
    SAFE_ZONE_DRAW_COLOR = (255, 0, 0)

# ===================== tracking memory parameters =====================
last_tracked_cx = None
last_tracked_cy = None
last_tracked_type = None

# 追踪惯性参数
TRACK_INERTIA_BONUS = 20  # 惯性加分，必须足够大，碾压面积和中心偏离的分差
TRACK_MATCH_DIST = 100       # 曼哈顿像素距离阈值：两帧之间球移动不超过100像素，认作同一个球

# ===================== uart protocol =====================
# 串口通信协议参数。
# 下位机协议：55 0D sign angle pos_x pos_y CRC16_L CRC16_H AA
# sign = vision_mode * 100 + cmd * 10 + grip_action
# CRC16 使用下位机 Buffer_usart.c 的范围：frame[0] ~ frame[9]。
UART_SEND_MIN_INTERVAL_MS = 30
last_sent_payload = None
last_send_time = 0
VISION_POS_MAX = 4095
UART_FRAME_LENGTH = 13
UART_FIELD_MAX = 65535


class MoveCmd:
    # 下位机约定的运动指令枚举。D0 字节通常放这里的值。
    STOP = 0x00
    FORWARD = 0x01
    BACKWARD = 0x02
    TURN_LEFT = 0x03
    TURN_RIGHT = 0x04


class GripAction:
    # 夹爪动作枚举。D2 字节通常放这里的值。
    GRAB = 0x01
    RELEASE = 0x02


class VisionMode:
    # D1 = vision_mode:
    # 0 NONE, 1 BALL_XY, 2 SAFE_XY.
    # Keep the original numeric values for lower MCU compatibility.
    NONE = 0x00
    BALL_XY = 0x01
    SAFE_XY = 0x02


# ===================== states =====================
# 主状态机定义。整个程序每一帧都会根据 ROBOT_STATE 进入不同逻辑分支。
STATE_SEARCH = 0
STATE_TRACK = 1
STATE_GRAB = 2
STATE_TRANSPORT = 3
STATE_UNLOAD = 4

# ===================== transport fallback parameters =====================
# TRANSPORT 中安全区短暂丢失时的搜索动作。
# 需要根据安全区识别稳定性和底盘转向速度现场调参。
TRANSPORT_LOST_SEARCH_CMD = MoveCmd.TURN_LEFT

# TRANSPORT 中连续丢失安全区多少帧后开始搜索。
# 需要根据识别帧率和误检/漏检情况现场调参。
TRANSPORT_LOST_MAX = 3

# ===================== globals =====================
# 当前主状态机状态，贯穿 SEARCH/TRACK/GRAB/TRANSPORT/UNLOAD。
ROBOT_STATE = STATE_SEARCH

# 是否已经完成第一次己方普通球运输，用于目标优先级切换。
FIRST_TARGET_DONE = False

# 已完成运输次数，仅用于流程统计和 debug。
TRANSPORT_COUNT = 0

# 当前夹持目标类型，用于 debug 和卸载后恢复。
CURRENT_HOLDING_TYPE = "NONE"

# GRAB 阶段锁定的目标类型，避免闭爪期间目标丢失后类型未知。
LOCKED_TARGET_TYPE = None

# TRACK -> GRAB 的 x/y 到位连续计数。
grab_ready_count = 0

# GRAB 阶段夹爪 ROI 连续确认计数。
grab_roi_ready_count = 0

# GRAB 阶段开始闭爪的时间戳，单位 ms。
grab_close_start_time = 0

# UNLOAD 阶段保留计数，当前主要用于状态恢复。
unload_ready_count = 0

# SEARCH -> TRACK 的连续锁定目标计数。
search_target_ready_count = 0

# UART 句柄，由 K230 UART 初始化后赋值。
uart = None

# GRAB/UNLOAD 等定时动作的起始时间戳，单位 ms。
action_timer = 0

# 【新增】SEARCH 致盲冷却计时器
search_cooldown_timer = 0

# TRANSPORT 安全区丢失计数，用于决定停车等待或转向搜索。
transport_lost_count = 0

# TRANSPORT 当前阶段：XY_APPROACH 或 SELF_ALIGN。
transport_phase = TRANSPORT_PHASE_XY_APPROACH

# TRANSPORT -> SELF_ALIGN 的 x/y 到位连续计数。
transport_xy_ready_count = 0

# SELF_ALIGN 进入时间，用于最小等待和超时兜底。
transport_align_start_time = 0

# SELF_ALIGN 中画面稳定连续计数，用于进入 UNLOAD。
transport_align_stable_count = 0

# SELF_ALIGN 中是否观察到过明显画面运动，仅用于 debug。
transport_align_seen_motion = False

# SELF_ALIGN 上一帧安全区中心，用于估计画面变化和稳定性。
transport_last_safe_x = None
transport_last_safe_y = None


def UART_init():
    # 初始化 K230 的 FPIOA 引脚复用，把 11/12 号引脚配置为 UART2 的 TX/RX。
    # 如果实际接线换了引脚，需要同步修改这里。
    fpioa = FPIOA()
    fpioa.set_function(11, FPIOA.UART2_TXD)
    fpioa.set_function(12, FPIOA.UART2_RXD)


def calculate_crc16(data, start_byte, end_byte):
    crc = 0xFFFF
    for i in range(start_byte, end_byte + 1):
        crc ^= data[i]
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def clamp_u16(value):
    value = int(value)
    if value < 0:
        value = 0
    if value > UART_FIELD_MAX:
        value = UART_FIELD_MAX
    return value


def pack_sign(cmd, vision_mode, grip_action):
    return int(vision_mode) * 100 + int(cmd) * 10 + int(grip_action)


def build_frame(payload):
    # 根据 STM32 下位机 Buffer_usart.c 的 DeviceControlPacket 打包。
    # payload 为 4 个 16-bit 字段：sign, angle, pos_x, pos_y。
    frame = bytearray(UART_FRAME_LENGTH)
    frame[0] = 0x55
    frame[1] = UART_FRAME_LENGTH

    for i in range(4):
        value = clamp_u16(payload[i])
        frame[2 + i * 2] = (value >> 8) & 0xFF
        frame[3 + i * 2] = value & 0xFF

    crc16 = calculate_crc16(frame, 0, 9)
    frame[10] = crc16 & 0xFF
    frame[11] = (crc16 >> 8) & 0xFF
    frame[12] = 0xAA
    return frame


def send_payload(payload):
    # 直接发送一帧数据，不做频率限制，也不判断是否和上一帧相同。
    uart.write(build_frame(payload))


def clamp_vision_pos(value):
    value = int(value)
    if value < 0:
        value = 0
    if value > VISION_POS_MAX:
        value = VISION_POS_MAX
    return value


def build_fixed5_payload(cmd, vision_mode, grip_action, pos_x_value=0, pos_y_value=0, angle_value=0):
    sign = pack_sign(cmd, vision_mode, grip_action)
    angle = 0
    pos_x = 0
    pos_y = 0

    if vision_mode == VisionMode.BALL_XY:
        pos_x = clamp_vision_pos(pos_x_value)
        pos_y = clamp_vision_pos(pos_y_value)
    elif vision_mode == VisionMode.SAFE_XY:
        angle = clamp_u16(angle_value)
        pos_x = clamp_vision_pos(pos_x_value)
        pos_y = clamp_vision_pos(pos_y_value)
    return (sign, angle, pos_x, pos_y)


def make_normal_payload(cmd, grip_action):
    # 普通运动/夹爪控制帧。没有要发送的视觉 x 坐标，所以 vision_position 填 0。
    return build_fixed5_payload(cmd, VisionMode.NONE, grip_action)


def make_track_payload(grip_action, target_x, target_y):
    # TRACK 阶段专用帧。
    # D0 固定为 FORWARD，同时把目标球中心坐标放入 pos_x/pos_y 字段。
    return build_fixed5_payload(MoveCmd.FORWARD, VisionMode.BALL_XY, grip_action, target_x, target_y)


# 【修改点】：增加 cmd 参数，解除固定 FORWARD 的限制
def make_transport_xy_payload(cmd, grip_action, safe_x, safe_y, align_flag=0):
    return build_fixed5_payload(
        cmd,
        VisionMode.SAFE_XY,
        grip_action,
        safe_x,
        safe_y,
        align_flag,
    )

def send_payload_if_needed(payload):
    global last_sent_payload, last_send_time

    now = time.ticks_ms()
    changed = last_sent_payload != payload
    timeout = time.ticks_diff(now, last_send_time) >= UART_SEND_MIN_INTERVAL_MS

    if changed or timeout:
        send_payload(payload)
        last_sent_payload = tuple(payload)
        last_send_time = now


def draw_label(img, x, y, text, color):
    # 在图像上绘制带黑色阴影的文字，提升调试显示的可读性。
    char_size = 32
    img.draw_string_advanced(x + 1, y + 1, char_size, text, (0, 0, 0))
    img.draw_string_advanced(x, y, char_size, text, color)


def get_display_label(label):
    name_map = {
        "redball": "RED",
        "blueball": "BLUE",
        "blackball": "CORE",
        "yellowball": "DANGER",
        "redboard": "REDBOARD",
        "blueboard": "BLUEBOARD",
        "redsection": "REDSECTION",
        "bluesection": "BLUESECTION",
        "shizi": "SHIZI",
        "redcube": "REDCUBE",
        "bluecube": "BLUECUBE",
        "greenzhu": "GREENZHU",
        "yellowtai": "YELLOWTAI",
    }
    return name_map.get(label, str(label))


def is_target_in_grip_roi(target):
    if target is None:
        return False

    x, y, w, h = GRIP_ROI
    tx = target["cx"]
    ty = target["cy"]

    in_x = x <= tx <= x + w
    in_y = y <= ty <= y + h
    area_ok = target["area"] >= GRIP_BALL_MIN_AREA

    return in_x and in_y and area_ok


def parse_yolo_results(res):
    dets = []
    if res is None:
        return dets

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

        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w == 0 or h == 0:
            return

        if score is None:
            score = 0.0

        cx = x1 + w // 2
        cy = y1 + h // 2
        area = w * h

        dets.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "w": w,
            "h": h,
            "cx": cx,
            "cy": cy,
            "area": area,
            "score": float(score),
            "class_id": class_id,
            "label": label,
            "rect": (x1, y1, w, h),
        })

    if isinstance(res, (tuple, list)) and len(res) == 3:
        boxes = res[0]
        class_ids = res[1]
        scores = res[2]
        if isinstance(boxes, (tuple, list)) and isinstance(class_ids, (tuple, list)):
            count = min(len(boxes), len(class_ids))
            if isinstance(scores, (tuple, list)):
                count = min(count, len(scores))
            for i in range(count):
                box = boxes[i]
                if box is None or len(box) < 4:
                    continue
                score = scores[i] if isinstance(scores, (tuple, list)) else 0.0
                class_id = class_ids[i]
                append_det(box[0], box[1], box[2], box[3], score, class_id, None)
            return dets

    for entry in res:
        if entry is None:
            continue

        x1 = y1 = x2 = y2 = score = class_id = label = None

        if isinstance(entry, dict):
            x1 = entry.get("x1") if entry.get("x1") is not None else entry.get("x")
            y1 = entry.get("y1") if entry.get("y1") is not None else entry.get("y")
            x2 = entry.get("x2") if entry.get("x2") is not None else None
            y2 = entry.get("y2") if entry.get("y2") is not None else None
            score = entry.get("score") if entry.get("score") is not None else entry.get("confidence") if entry.get("confidence") is not None else entry.get("prob")
            class_id = entry.get("class_id") if entry.get("class_id") is not None else entry.get("classid") if entry.get("classid") is not None else entry.get("classId")
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
            x1 = getattr(entry, "x1", None) if getattr(entry, "x1", None) is not None else getattr(entry, "x", None)
            y1 = getattr(entry, "y1", None) if getattr(entry, "y1", None) is not None else getattr(entry, "y", None)
            x2 = getattr(entry, "x2", None) if getattr(entry, "x2", None) is not None else getattr(entry, "w", None)
            y2 = getattr(entry, "y2", None) if getattr(entry, "y2", None) is not None else getattr(entry, "h", None)
            score = getattr(entry, "score", None) if getattr(entry, "score", None) is not None else getattr(entry, "confidence", None)
            class_id = getattr(entry, "class_id", None) if getattr(entry, "class_id", None) is not None else getattr(entry, "classid", None)
            label = getattr(entry, "label", None)
            if x2 is None and getattr(entry, "w", None) is not None and x1 is not None:
                x2 = x1 + getattr(entry, "w", None)
            if y2 is None and getattr(entry, "h", None) is not None and y1 is not None:
                y2 = y1 + getattr(entry, "h", None)

        append_det(x1, y1, x2, y2, score, class_id, label)

    return dets


def reset_transport_controller():
    global transport_lost_count, transport_phase
    global transport_xy_ready_count, transport_align_start_time
    global transport_align_stable_count, transport_align_seen_motion
    global transport_last_safe_x, transport_last_safe_y

    transport_lost_count = 0
    transport_phase = TRANSPORT_PHASE_XY_APPROACH
    transport_xy_ready_count = 0
    transport_align_start_time = 0
    transport_align_stable_count = 0
    transport_align_seen_motion = False
    transport_last_safe_x = None
    transport_last_safe_y = None


def get_yolo_ball_targets(detections):
    targets = []
    type_map = {
        "redball": "NORMAL_RED",
        "blueball": "NORMAL_BLUE",
        "blackball": "CORE",
        "yellowball": "DANGER",
    }
    for det in detections:
        if det["label"] not in type_map:
            continue
        if det["area"] < MIN_DRAW_AREA or det["area"] > MAX_DRAW_AREA:
            continue
        targets.append({
            "cx": det["cx"],
            "cy": det["cy"],
            "area": det["area"],
            "type": type_map[det["label"]],
            "rect": det["rect"],
            "label": det["label"],
        })
    return targets


def detect_safe_zone_from_yolo(detections):
    safe_label = "blueboard" if TEAM_COLOR == "BLUE" else "redboard"
    best_safe = None
    for det in detections:
        if det["label"] != safe_label:
            continue
        if det["area"] < SAFE_ZONE_DETECT_MIN_AREA:
            continue
        if best_safe is None or det["area"] > best_safe["bbox_area"]:
            best_safe = {
                "cx": det["cx"],
                "cy": det["cy"],
                "bbox_area": det["area"],
                "rect": det["rect"],
                "outer_rect": det["rect"],
                "label": det["label"],
            }
    return best_safe


def get_sections_from_yolo(detections):
    sections = []
    for det in detections:
        if det["label"] in ["redsection", "bluesection", "shizi", "redcube", "bluecube", "greenzhu", "yellowtai"]:
            sections.append(det)
    return sections


def draw_yolo_detections(img, detections, control_target, safe_blob):
    if detections is None:
        return

    for det in detections:
        if safe_blob is not None and det["rect"] == safe_blob["rect"]:
          continue
        color = (200, 200, 200)
        if det["label"] == "redball":
            color = (255, 0, 0)
        elif det["label"] == "blueball":
            color = (0, 0, 255)
        elif det["label"] == "blackball":
            color = (255, 255, 255)
        elif det["label"] == "yellowball":
            color = (255, 255, 0)
        elif det["label"] in ["redboard", "redsection"]:
            color = (255, 0, 0)
        elif det["label"] in ["blueboard", "bluesection"]:
            color = (0, 0, 255)

        img.draw_rectangle(det["rect"], color=color, thickness=2)
        img.draw_cross(det["cx"], det["cy"], color=color, size=10)
        draw_label(img, det["x1"], det["y1"] - 24, get_display_label(det["label"]), color)

    if safe_blob is not None:
        img.draw_rectangle(safe_blob["outer_rect"], color=SAFE_ZONE_DRAW_COLOR, thickness=3)
        draw_label(img, safe_blob["x1"] if "x1" in safe_blob else safe_blob["outer_rect"][0],
                   safe_blob["y1"] - 24 if "y1" in safe_blob else safe_blob["outer_rect"][1] - 24,
                   "SAFE_ZONE", SAFE_ZONE_DRAW_COLOR)

    if control_target is not None:
        img.draw_cross(control_target["cx"], control_target["cy"], color=(0, 255, 255), size=12)
        img.draw_rectangle(control_target["rect"], color=(0, 255, 255), thickness=2)


def get_priority_target(targets_list, detections):
    global FIRST_TARGET_DONE

    valid_targets = []

    weights = {
        "CORE": WEIGHT_CORE,
        "DANGER": WEIGHT_DANGER,
    }

    if TEAM_COLOR == "RED":
        weights["NORMAL_RED"] = WEIGHT_NORMAL
    if TEAM_COLOR == "BLUE":
        weights["NORMAL_BLUE"] = WEIGHT_NORMAL

    filter_labels = ["blueboard", "redboard"]

    # 1. 目标入围筛选阶段：
    for t in targets_list:
        in_safe_zone = False
        tx, ty = t["cx"], t["cy"]
        for det in detections:
            if det["label"] in filter_labels:
                pad = 30 # 外扩容差
                if (det["x1"] - pad) <= tx <= (det["x2"] + pad) and (det["y1"] - pad) <= ty <= (det["y2"] + pad):
                    in_safe_zone = True
                    break

        if in_safe_zone:
            continue

        if not FIRST_TARGET_DONE:
            if TEAM_COLOR == "RED" and t["type"] == "NORMAL_RED":
                valid_targets.append(t)
            elif TEAM_COLOR == "BLUE" and t["type"] == "NORMAL_BLUE":
                valid_targets.append(t)
        else:
            if t["type"] in weights:
                valid_targets.append(t)

    if not valid_targets:
        return None

    # 2. 类别提取：找出当前视野里，具有最高权重的球的类别
    best_type = max(valid_targets, key=lambda t: weights.get(t["type"], 0))["type"]
    candidates = [t for t in valid_targets if t["type"] == best_type]

    # 3. 距离与惯性打分：在同类球中挑选一个最优的
    def score_target(t):
        center_penalty = abs(t["cx"] - CENTER_X) * CENTER_SCORE_PENALTY
        base_score = t["area"] - center_penalty

        if last_tracked_cx is not None and last_tracked_cy is not None:
            if t["type"] == last_tracked_type:
                dist = abs(t["cx"] - last_tracked_cx) + abs(t["cy"] - last_tracked_cy)
                if dist < TRACK_MATCH_DIST:
                    base_score += TRACK_INERTIA_BONUS

        return base_score

    return max(candidates, key=score_target)


try:
    # ===================== 初始化阶段 =====================
    pl = PipeLine(rgb888p_size=rgb888p_size, display_mode=display_mode, display_size=display_size)
    sensor = Sensor(id=sensor_id)
    pl.create(sensor)
    display_size = pl.get_display_size()

    yolo = YOLOv5(
        task_type="detect",
        mode="video",
        kmodel_path=KMODEL_PATH,
        labels=LABELS,
        rgb888p_size=rgb888p_size,
        model_input_size=MODEL_INPUT_SIZE,
        display_size=display_size,
        conf_thresh=confidence_threshold,
        nms_thresh=nms_threshold,
        max_boxes_num=max_boxes_num,
        debug_mode=0,
    )
    yolo.config_preprocess()

    UART_init()
    uart = UART(
        UART.UART2,
        baudrate=115200,
        bits=UART.EIGHTBITS,
        parity=UART.PARITY_NONE,
        stop=UART.STOPBITS_ONE,
    )

    send_payload_if_needed(make_normal_payload(MoveCmd.STOP, GripAction.RELEASE))
    print("system start... ({})".format(VERSION))
    print(">>> full FSM enabled: SEARCH -> TRACK -> GRAB -> TRANSPORT -> UNLOAD")

    action_timer = time.ticks_ms()
    STATE_NAMES = {
        STATE_SEARCH: "SEARCH",
        STATE_TRACK: "TRACK",
        STATE_GRAB: "GRAB",
        STATE_TRANSPORT: "TRANSPORT",
        STATE_UNLOAD: "UNLOAD",
    }
    LAST_ROBOT_STATE = -1
    debug_print_count = 0

    while True:
        # ===================== 主循环 =====================
        os.exitpoint()
        img = pl.get_frame()
        osd_img = pl.osd_img
        try:
            osd_img.clear()
        except Exception:
            pass

        res = yolo.run(img)
        if debug_print_count < 3:
            print("YOLO raw result:", res)
        debug_print_count += 1
        if ROBOT_STATE != LAST_ROBOT_STATE:
            print("\n>>> [STATE] ---> {}".format(STATE_NAMES.get(ROBOT_STATE, "UNKNOWN")))
            LAST_ROBOT_STATE = ROBOT_STATE

        dets = parse_yolo_results(res)
        safe_blob = detect_safe_zone_from_yolo(dets)
        safe_detected = safe_blob is not None

        all_targets = get_yolo_ball_targets(dets)
        control_target = get_priority_target(all_targets, dets)
        sections = get_sections_from_yolo(dets)

        # ==========================================
        # 更新追踪记忆
        if control_target is not None:
            last_tracked_cx = control_target["cx"]
            last_tracked_cy = control_target["cy"]
            last_tracked_type = control_target["type"]
        elif ROBOT_STATE == STATE_SEARCH:
            last_tracked_cx = None
            last_tracked_cy = None
            last_tracked_type = None
        # ==========================================

        draw_yolo_detections(osd_img, dets, control_target, safe_blob)
        if debug_print_count <= 3:
            print("YOLO targets:", len(all_targets), "safe:", safe_detected, "sections:", len(sections))

        cmd = MoveCmd.STOP
        grip = GripAction.RELEASE
        payload = make_normal_payload(cmd, grip)
        transport_debug_safe_x = None
        transport_debug_safe_y = None
        transport_debug_box_area = None
        transport_debug_xy_ready_count = transport_xy_ready_count
        transport_debug_align_stable_count = transport_align_stable_count
        transport_debug_align_flag = TRANSPORT_ALIGN_FLAG_NONE

        if ROBOT_STATE == STATE_SEARCH:
            grip = GripAction.RELEASE
            grab_ready_count = 0
            grab_roi_ready_count = 0
            grab_close_start_time = 0
            unload_ready_count = 0
            reset_transport_controller()

            is_cooldown = time.ticks_diff(time.ticks_ms(), search_cooldown_timer) < SEARCH_COOLDOWN_MS

            if control_target and control_target["area"] >= SEARCH_LOCK_MIN_AREA and not is_cooldown:
                search_target_ready_count += 1
            else:
                search_target_ready_count = 0

            if search_target_ready_count >= SEARCH_ENTER_TRACK_FRAMES:
                ROBOT_STATE = STATE_TRACK
                cmd = MoveCmd.STOP
                payload = make_normal_payload(cmd, grip)
                print(">>> enter TRACK: type={} cx={} cy={} area={}".format(
                    control_target["type"], control_target["cx"], control_target["cy"], control_target["area"]
                ))
            else:
                cmd = MoveCmd.TURN_LEFT
                payload = make_normal_payload(cmd, grip)

        elif ROBOT_STATE == STATE_TRACK:
            grip = GripAction.RELEASE

            if control_target:
                transport_lost_count = 0

                ball_x_ok = abs(control_target["cx"] - CENTER_X) <= TRACK_X_DEAD_ZONE
                ball_y_ok = control_target["cy"] >= TRACK_GRAB_TARGET_Y

                if ball_x_ok and ball_y_ok:
                    grab_ready_count += 1
                else:
                    grab_ready_count = 0

                if grab_ready_count >= GRAB_CONFIRM_FRAMES:
                    ROBOT_STATE = STATE_GRAB
                    LOCKED_TARGET_TYPE = control_target["type"]
                    search_target_ready_count = 0
                    grab_roi_ready_count = 0
                    grab_close_start_time = 0
                    action_timer = time.ticks_ms()
                    cmd = MoveCmd.STOP
                    payload = make_normal_payload(cmd, grip)
                    print(">>> prepare GRAB: type={} cx={} cy={} area={}".format(
                        LOCKED_TARGET_TYPE, control_target["cx"], control_target["cy"], control_target["area"]
                    ))
                else:
                    payload = make_track_payload(
                        grip_action=grip,
                        target_x=control_target["cx"],
                        target_y=control_target["cy"],
                    )
            else:
                grab_ready_count = 0
                transport_lost_count += 1

                if transport_lost_count <= TRANSPORT_LOST_MAX:
                    cmd = MoveCmd.STOP
                    payload = make_normal_payload(cmd, grip)
                else:
                    ROBOT_STATE = STATE_SEARCH
                    transport_lost_count = 0
                    search_target_ready_count = 0
                    cmd = MoveCmd.TURN_LEFT
                    payload = make_normal_payload(cmd, grip)

        elif ROBOT_STATE == STATE_GRAB:
            now_ms = time.ticks_ms()

            if grab_close_start_time != 0:
                cmd = MoveCmd.STOP
                grip = GripAction.GRAB

                if time.ticks_diff(now_ms, grab_close_start_time) >= GRIP_CLOSE_TIME_MS:
                    TRANSPORT_COUNT += 1
                    CURRENT_HOLDING_TYPE = LOCKED_TARGET_TYPE if LOCKED_TARGET_TYPE else (
                        control_target["type"] if control_target else "UNKNOWN"
                    )
                    if (not FIRST_TARGET_DONE) and CURRENT_HOLDING_TYPE in ["NORMAL_RED", "NORMAL_BLUE"]:
                        FIRST_TARGET_DONE = True

                    ROBOT_STATE = STATE_TRANSPORT
                    search_target_ready_count = 0
                    grab_ready_count = 0
                    grab_roi_ready_count = 0
                    grab_close_start_time = 0
                    unload_ready_count = 0
                    reset_transport_controller()
                    action_timer = now_ms
                    print(">>> [视觉确认抓取完成] 当前持有: {}, 已运输次数: {}".format(CURRENT_HOLDING_TYPE, TRANSPORT_COUNT))

                payload = make_normal_payload(cmd, grip)

            elif control_target:
                grip = GripAction.RELEASE

                if is_target_in_grip_roi(control_target):
                    grab_roi_ready_count += 1
                else:
                    grab_roi_ready_count = 0

                if grab_roi_ready_count < GRIP_BALL_CONFIRM_FRAMES:
                    payload = make_track_payload(
                        grip_action=grip,
                        target_x=control_target["cx"],
                        target_y=control_target["cy"],
                    )
                else:
                    cmd = MoveCmd.STOP
                    grip = GripAction.GRAB
                    grab_close_start_time = now_ms
                    payload = make_normal_payload(cmd, grip)

            else:
                grab_roi_ready_count = 0
                grab_close_start_time = 0
                ROBOT_STATE = STATE_SEARCH
                cmd = MoveCmd.TURN_LEFT
                grip = GripAction.RELEASE
                payload = make_normal_payload(cmd, grip)
                print(">>> GRAB failed: target lost, back to SEARCH")

        elif ROBOT_STATE == STATE_TRANSPORT:
            grip = GripAction.GRAB
            now_ms = time.ticks_ms()

            if safe_detected:
                transport_lost_count = 0

                safe_x = safe_blob["cx"]
                safe_y = safe_blob["cy"]
                safe_box_area = safe_blob["bbox_area"]
                err_x = safe_x - CENTER_X
                transport_debug_safe_x = safe_x
                transport_debug_safe_y = safe_y
                transport_debug_box_area = safe_box_area

                safe_x_ok = abs(err_x) <= TRANSPORT_X_DEAD_ZONE
                safe_y_ok = safe_y >= TRANSPORT_SAFE_TARGET_Y

                if transport_phase == TRANSPORT_PHASE_XY_APPROACH:
                    transport_debug_align_flag = TRANSPORT_ALIGN_FLAG_NONE
                    # 【修改点】：靠近阶段仍然传入 FORWARD（保持生成211）
                    payload = make_transport_xy_payload(
                        MoveCmd.FORWARD,
                        GripAction.GRAB,
                        safe_x,
                        safe_y,
                        TRANSPORT_ALIGN_FLAG_NONE,
                    )

                    if safe_x_ok and safe_y_ok:
                        transport_xy_ready_count += 1
                    else:
                        transport_xy_ready_count = 0

                    if transport_xy_ready_count >= TRANSPORT_XY_CONFIRM_FRAMES:
                        transport_phase = TRANSPORT_PHASE_SELF_ALIGN
                        transport_align_start_time = now_ms
                        transport_align_stable_count = 0
                        transport_align_seen_motion = False
                        transport_last_safe_x = safe_x
                        transport_last_safe_y = safe_y
                        transport_debug_align_flag = TRANSPORT_ALIGN_FLAG_START

                        # 【修改点】：进入自校正瞬间，立即传入 STOP（生成 201）
                        payload = make_transport_xy_payload(
                            MoveCmd.STOP,
                            GripAction.GRAB,
                            safe_x,
                            safe_y,
                            TRANSPORT_ALIGN_FLAG_START,
                        )

                        print(">>> TRANSPORT enter SELF_ALIGN: send align flag")

                    print("TRANSPORT XY safe_x={} safe_y={} err={} box={} x_ok={} y_ok={} xy_cnt={}".format(
                        safe_x, safe_y, err_x, safe_box_area, safe_x_ok, safe_y_ok,
                        transport_xy_ready_count
                    ))

                elif transport_phase == TRANSPORT_PHASE_SELF_ALIGN:
                    transport_debug_align_flag = TRANSPORT_ALIGN_FLAG_START
                    # 【修改点】：自校正全过程传入 STOP（生成 201）
                    payload = make_transport_xy_payload(
                        MoveCmd.STOP,
                        GripAction.GRAB,
                        safe_x,
                        safe_y,
                        TRANSPORT_ALIGN_FLAG_START,
                    )

                    dx = 0 if transport_last_safe_x is None else abs(safe_x - transport_last_safe_x)
                    dy = 0 if transport_last_safe_y is None else abs(safe_y - transport_last_safe_y)

                    if dx >= ALIGN_MOVE_X_TH or dy >= ALIGN_MOVE_Y_TH:
                        transport_align_seen_motion = True

                    frame_stable = dx <= ALIGN_STABLE_X_TH and dy <= ALIGN_STABLE_Y_TH
                    xy_still_ok = safe_x_ok and safe_y_ok
                    waited_enough = time.ticks_diff(now_ms, transport_align_start_time) >= ALIGN_WAIT_MIN_MS
                    timeout = time.ticks_diff(now_ms, transport_align_start_time) >= ALIGN_WAIT_TIMEOUT_MS

                    if waited_enough and frame_stable and xy_still_ok:
                        transport_align_stable_count += 1
                    else:
                        transport_align_stable_count = 0

                    if transport_align_stable_count >= ALIGN_STABLE_CONFIRM_FRAMES or timeout:
                        # 【修改点】：结束自校正时，依然保持 STOP（生成 201）
                        payload = make_transport_xy_payload(
                            MoveCmd.STOP,
                            GripAction.GRAB,
                            safe_x,
                            safe_y,
                            TRANSPORT_ALIGN_FLAG_NONE,
                        )
                        send_payload(payload)
                        last_sent_payload = tuple(payload)
                        last_send_time = time.ticks_ms()

                        ROBOT_STATE = STATE_UNLOAD
                        action_timer = now_ms
                        reset_transport_controller()
                        print(">>> prepare UNLOAD: self align stable or timeout")

                    transport_last_safe_x = safe_x
                    transport_last_safe_y = safe_y

                    print("TRANSPORT SELF_ALIGN safe_x={} safe_y={} dx={} dy={} stable_cnt={} seen_motion={} timeout={}".format(
                        safe_x, safe_y, dx, dy, transport_align_stable_count,
                        transport_align_seen_motion, timeout
                    ))

                else:
                    reset_transport_controller()
                    payload = make_transport_xy_payload(
                        MoveCmd.FORWARD,
                        GripAction.GRAB,
                        safe_x,
                        safe_y,
                        TRANSPORT_ALIGN_FLAG_NONE,
                    )

                transport_debug_xy_ready_count = transport_xy_ready_count
                transport_debug_align_stable_count = transport_align_stable_count
            else:
                unload_ready_count = 0
                transport_xy_ready_count = 0
                transport_align_stable_count = 0
                transport_lost_count += 1

                if transport_lost_count >= TRANSPORT_LOST_MAX:
                    transport_phase = TRANSPORT_PHASE_XY_APPROACH
                    cmd = TRANSPORT_LOST_SEARCH_CMD
                else:
                    cmd = MoveCmd.STOP

                payload = make_normal_payload(cmd, grip)

                print("TRANSPORT safe_detected=0 lost_cnt={} cmd={}".format(
                    transport_lost_count, cmd
                ))

        elif ROBOT_STATE == STATE_UNLOAD:
            elapsed = time.ticks_diff(time.ticks_ms(), action_timer)

            if elapsed < 500:
                cmd = MoveCmd.STOP
                grip = GripAction.GRAB
            elif elapsed < 5000:
                cmd = MoveCmd.FORWARD
                grip = GripAction.RELEASE
            elif elapsed < 9000:
                cmd = MoveCmd.BACKWARD
                grip = GripAction.RELEASE
            else:
                CURRENT_HOLDING_TYPE = "NONE"
                LOCKED_TARGET_TYPE = None
                unload_ready_count = 0
                search_target_ready_count = 0
                ROBOT_STATE = STATE_SEARCH
                reset_transport_controller()

                search_cooldown_timer = time.ticks_ms()

                cmd = MoveCmd.STOP
                grip = GripAction.RELEASE
                print(">>> unload complete: back to SEARCH with Cooldown")

            payload = make_normal_payload(cmd, grip)

        center_y_draw = SENSOR_HEIGHT // 2
        osd_img.draw_rectangle(GRIP_ROI, color=(0, 255, 0), thickness=2)
        draw_label(osd_img, GRIP_ROI[0], GRIP_ROI[1] - 25, "GRIP_ROI", (0, 255, 0))
        osd_img.draw_cross(CENTER_X, center_y_draw, color=(0, 255, 0), size=12)
        draw_label(osd_img, CENTER_X + 10, center_y_draw + 10, "CENTER_X", (0, 255, 0))

        # ==========================================================
        # OSD 屏幕左上角全局状态与真实 Sign 值显示
        current_state_name = STATE_NAMES.get(ROBOT_STATE, "UNKNOWN")
        actual_sign = payload[0]  # 取出当前帧实际要发送的 sign 值
        draw_label(osd_img, 10, 10, "STATE:{} SIGN:{}".format(current_state_name, actual_sign), (255, 255, 0))

        if control_target:
            osd_img.draw_line(CENTER_X, center_y_draw, control_target["cx"], control_target["cy"], color=(0, 255, 255), thickness=2)
            err_x = control_target["cx"] - CENTER_X
            draw_label(osd_img, 10, 42, "EX:{} TX:{} TY:{}".format(err_x, control_target["cx"], control_target["cy"]), (0, 255, 255))
            draw_label(osd_img, 10, 74, "T_AREA:{} T_TYPE:{}".format(control_target["area"], get_display_label(control_target["label"])), (0, 255, 255))

        if ROBOT_STATE == STATE_TRANSPORT:
            phase_name = "XY" if transport_phase == TRANSPORT_PHASE_XY_APPROACH else "SELF_ALIGN"
            draw_label(osd_img, 10, 106, "TRANSPORT_PHASE:{}".format(phase_name), (255, 0, 0))
            draw_label(osd_img, 10, 138, "safe_x:{}".format(transport_debug_safe_x), (255, 0, 0))
            draw_label(osd_img, 10, 170, "safe_y:{}".format(transport_debug_safe_y), (255, 0, 0))
            draw_label(osd_img, 10, 202, "safe_box_area:{}".format(transport_debug_box_area), (255, 0, 0))
            draw_label(osd_img, 10, 234, "xy_ready_count:{}".format(transport_debug_xy_ready_count), (255, 0, 0))
            draw_label(osd_img, 10, 266, "align_stable_count:{}".format(transport_debug_align_stable_count), (255, 0, 0))
            draw_label(osd_img, 10, 298, "align_flag:{}".format(transport_debug_align_flag), (255, 0, 0))
        # ==========================================================

        send_payload_if_needed(payload)
        pl.show_image()
        time.sleep_ms(10)

except Exception as e:
    print("Error:", e)
finally:
    try:
        if sensor:
            sensor.stop()
    except Exception:
        pass
    try:
        if yolo is not None:
            yolo.deinit()
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
