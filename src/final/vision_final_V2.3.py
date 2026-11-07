# ============================================================
# 工创赛救援小车 K230 决赛视觉上位机程序 V2.3
# YOLO 识别红/蓝球与安全区，CV2 HSV 识别黑方块与黄色三棱锥。
# CV 使用独立抓取 ROI；框底进入该区域连续3帧后闭爪并直接运输。
# ============================================================

import time
import os
import image
import cv2
from media.sensor import *
from media.display import *
from media.media import MediaManager
from machine import UART
from machine import FPIOA
from libs.YOLO import YOLOv5
from libs.Utils import *
import ulab.numpy as np
import gc

VERSION = "Final Vision v2.3 independent CV grab ROI"

# ===================== hardware =====================
# 摄像头与图像尺寸相关配置。
# 单摄像头三通道：CH0 显示、CH1 CV2 RGB888/HWC、CH2 YOLO RGBP888/CHW。
sensor_id = 1
sensor = None
yolo = None
osd_img = None
SENSOR_WIDTH = 800
SENSOR_HEIGHT = 480
CENTER_X = SENSOR_WIDTH // 2
CV_WIDTH = 320
CV_HEIGHT = 240
DISPLAY_CHN = CAM_CHN_ID_0
CV_CHN = CAM_CHN_ID_1
YOLO_CHN = CAM_CHN_ID_2

# v1.4 调度：CV 目标锁定后 CV 每帧运行，YOLO 每 2 帧运行一次。
# SEARCH 中若上一帧已有本方球，则本帧跳过 CV；TRANSPORT 只运行 YOLO。
YOLO_EVERY_N_FRAMES_WHEN_CV_LOCKED = 2

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

display_size = [800, 480]
rgb888p_size = [800, 480]
confidence_threshold = 0.4
nms_threshold = 0.45
max_boxes_num = 50

YOLO_NORMAL_BALL_LABELS = ("redball", "blueball")
YOLO_SAFE_LABELS = ("redboard", "blueboard")
# 决赛没有黑球/黄球；这两个模型输出只能辅助同色 CV 候选，不能独立成为控制目标。
YOLO_ASSIST_LABELS = ("blackball", "yellowball")

gc.collect()

# ===================== team / thresholds =====================
# 队伍颜色决定抓取顺序与目标优先级。
TEAM_COLOR = "BLUE"  # 比赛前修改队伍颜色： "RED" 或 "BLUE"
#TEAM_COLOR = "RED"
# ===================== final material priority =====================
# 硬优先级：本方红/蓝球 > 黑方块 CORE > 黄色三棱锥 DANGER。
# 第一件物料仍必须是本方普通球；对方普通球始终不进入控制候选。
CV_MATERIAL_ORDER = ("DANGER",)

# ===================== OpenCV HSV / geometry =====================
# 现场主要调整这两组 HSV。H=0..179，S/V=0..255。
BLACK_HSV_RANGES = (
    ((0, 0, 0), (179, 255, 70)),
)
YELLOW_HSV_RANGES = (
    ((18, 80, 60), (38, 255, 255)),
)

# 严格远场轮廓参数均基于 320x240 CV 图像。
CV_MATERIAL_CONFIG = {
    "CORE": {
        "color": "BLACK",
        "ranges": BLACK_HSV_RANGES,
        "draw_color": (255, 255, 255),
        "min_contour_area": 60,
        "max_contour_area": 22000,
        "min_width": 7,
        "min_height": 7,
        "min_aspect": 0.55,
        "max_aspect": 1.80,
        "min_fill": 0.48,
        "max_fill": 1.05,
        "min_local_contrast": 25.0,
        "use_local_contrast": True,
        "use_close": False,
        "aim_y_ratio": 0.5,
    },
    "DANGER": {
        "color": "YELLOW",
        "ranges": YELLOW_HSV_RANGES,
        "draw_color": (255, 255, 0),
        "min_contour_area": 60,
        "max_contour_area": 26000,
        "min_width": 7,
        "min_height": 7,
        "min_aspect": 0.45,
        "max_aspect": 2.20,
        "min_fill": 0.28,
        "max_fill": 1.05,
        "min_local_contrast": 0.0,
        "use_local_contrast": False,
        "use_close": True,
        "aim_y_ratio": 0.5,
    },
}

CV_BORDER_MARGIN = 2
CV_EXCLUDE_RECTS = ()
YOLO_OVERLAP_MARGIN = 12
CV_COVERED_RATIO_MIN = 0.20
SHOW_REJECTED_CV = True
MAX_REJECTED_DRAW = 8

# 上电后记录空闭爪 VERIFY_GRAB_ROI 的同色像素基线。
EMPTY_BASELINE_CALIBRATION_FRAMES = 20
STRICT_MATCH_MAX_DISTANCE = 180
STRICT_MATCH_MAX_X_DELTA = 120

# CV 闭爪备用确认参数，均基于 320x240 CV 图像中的 VERIFY_GRAB_ROI。
# 完整严格框仍在时优先使用严格框；只有严格框被夹爪遮挡时才使用这些原始 HSV 条件。
CV_VERIFY_CONFIG = {
    "CORE": {
        "min_contour_area": 500,
        "min_excess_pixels": 150,
    },
    "DANGER": {
        "min_contour_area": 150,
        "min_excess_pixels": 80,
    },
}

# ===================== target filter parameters =====================
# YOLO 目标球过滤参数，只用于筛掉太小/异常大的检测框，不作为到位判断。
# 需要根据摄像头视角和误检情况现场调参。
MIN_DRAW_AREA = 50
MAX_DRAW_AREA = 80000

# 目标优先级打分参数，偏离画面中心越远惩罚越大。
# 影响 SEARCH/TRACK 选择哪个球，可能需要根据赛场目标密度调参。
CENTER_SCORE_PENALTY = 60

# ===================== track/grab state parameters =====================
# TRACK 阶段球中心 x 允许误差，单位像素。
# 用于 TRACK -> GRAB，需要根据下位机 BALL_XY 横向 PID 稳定性现场调参。
TRACK_X_DEAD_ZONE = 80

# TRACK 阶段进入 GRAB 的球中心 y 目标值，单位像素。
# 必须和下位机 BALL_XY 模式下的 y 目标保持一致，否则上下位机状态判断会错位。
TRACK_GRAB_TARGET_Y = 420

# TRACK 阶段提前张开夹爪的球中心 y 阈值，单位像素。
# 该值只控制开爪时机，不改变 TRACK -> GRAB 与下位机 PID 共用的 420 目标值。
TRACK_GRIP_OPEN_TARGET_Y = 260

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
GRIP_ROI = (332, 415, 170, 62)

# CV 黑方块/黄锥专用抓取到位区域，与上面的 YOLO GRIP_ROI 完全独立。
# 判断点使用 CV 检测框的底部中心；现场只调整这里不会影响 YOLO 抓取。
CV_GRAB_ROI = (332, 390, 170, 80)

# GRAB 阶段 ROI 内球检测过滤面积，只用于误检过滤。
# 需要根据夹爪附近目标大小现场调参。
GRIP_BALL_MIN_AREA = 1000

# GRAB 阶段 ROI 连续确认帧数，抑制偶发误检。
# 需要根据画面稳定性现场调参。
GRIP_BALL_CONFIRM_FRAMES = 3

# 夹爪闭合保持时间，单位 ms。
# 需要根据舵机速度和实际抓取稳定性现场调参。
GRIP_CLOSE_TIME_MS = 1200

# VERIFY_GRAB 使用独立 ROI 对闭爪后的球进行二次确认。
# 当前默认复用 GRIP_ROI，现场可根据夹爪闭合后球在画面中的实际位置单独调整。
VERIFY_GRAB_ROI = GRIP_ROI

# VERIFY_GRAB 中球检测框的最小面积。
# 当前默认复用 GRIP_BALL_MIN_AREA，若闭爪遮挡导致检测框明显变小，可单独下调。
VERIFY_GRAB_BALL_MIN_AREA = GRIP_BALL_MIN_AREA

# 闭爪后必须连续看到锁定类型的小球多少帧，才确认抓取成功。
VERIFY_GRAB_CONFIRM_FRAMES = 2

# VERIFY_GRAB 最长等待时间，单位 ms。
# 超时仍未确认到球时，认为本次抓取失败，松爪并回到 SEARCH。
VERIFY_GRAB_TIMEOUT_MS = 1500

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

# ===================== transport xy parameters =====================
# TRANSPORT 阶段安全区中心 x 允许误差，单位像素。
# 用于 TRANSPORT -> UNLOAD，需要根据下位机 SAFE_XY 横向 PID 稳定性现场调参。
TRANSPORT_X_DEAD_ZONE = 150
# TRANSPORT 阶段安全区中心 y 到位目标值，单位像素。
# 必须和下位机 SAFE_XY 模式下的 y 目标保持一致。
TRANSPORT_SAFE_TARGET_Y = 290

# TRANSPORT -> UNLOAD 的 x/y 到位连续确认帧数。
# 需要根据安全区识别抖动和底盘响应速度现场调参。
TRANSPORT_XY_CONFIRM_FRAMES = 3

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
STATE_VERIFY_GRAB = 5
STATE_ESCAPE = 6

# ===================== transport fallback parameters =====================
# TRANSPORT 中安全区短暂丢失时的搜索动作。
# 需要根据安全区识别稳定性和底盘转向速度现场调参。
TRANSPORT_LOST_SEARCH_CMD = MoveCmd.TURN_LEFT

# TRANSPORT 中连续丢失安全区多少帧后开始搜索。
# 需要根据识别帧率和误检/漏检情况现场调参。
TRANSPORT_LOST_MAX = 3

# ===================== stuck escape parameters =====================
# 小车收到运动指令但画面持续不变达到该时间后，认为可能卡住。
# 现场只需要重点调整该时间和下面的画面差异阈值；ESCAPE 各阶段时间直接写在状态判断中。
STUCK_NO_MOTION_MS = 3000

# 80x48 灰度小图的平均帧差阈值。静止时若现场噪声较大可适当调高，误触发则调低。
STUCK_MOTION_SCORE_THRESHOLD = 3


class StuckDetector:
    # 把运动检测和挣脱方向等运行数据集中在一个对象中，避免增加大量独立全局变量。
    def __init__(self):
        self.previous_small = None
        self.last_sample_time = 0
        self.still_start_time = 0
        self.last_score = None
        self.available = True
        self.escape_turn_cmd = MoveCmd.TURN_RIGHT

    def reset_observation(self):
        self.previous_small = None
        self.last_sample_time = 0
        self.still_start_time = 0
        self.last_score = None

    def choose_escape_turn(self, trigger_move_cmd):
        # 转向时卡住就选择反方向；直行卡住时左右交替，减少重复顶住同一侧的概率。
        if trigger_move_cmd == MoveCmd.TURN_LEFT:
            self.escape_turn_cmd = MoveCmd.TURN_RIGHT
        elif trigger_move_cmd == MoveCmd.TURN_RIGHT:
            self.escape_turn_cmd = MoveCmd.TURN_LEFT
        elif self.escape_turn_cmd == MoveCmd.TURN_LEFT:
            self.escape_turn_cmd = MoveCmd.TURN_RIGHT
        else:
            self.escape_turn_cmd = MoveCmd.TURN_LEFT

    def update(self, source_img, now_ms, enabled):
        if not self.available:
            return False

        if not enabled:
            self.previous_small = None
            self.still_start_time = 0
            self.last_score = None
            return False

        # 每 200 ms 抽样一次，避免低分辨率运动检测明显影响 YOLO 帧率。
        if time.ticks_diff(now_ms, self.last_sample_time) < 200:
            return False

        self.last_sample_time = now_ms

        if source_img is None:
            return False

        try:
            # 必须在副本上灰度化，不能修改 CV2 使用的原始 RGB888 image.Image。
            current_small = source_img.to_grayscale(copy=True)
            if source_img.width() == CV_WIDTH and source_img.height() == CV_HEIGHT:
                current_small.mean_pool(4, 5)   # 320x240 -> 80x48
            else:
                current_small.mean_pool(10, 10) # 800x480 -> 80x48

            if self.previous_small is None:
                self.previous_small = current_small
                self.still_start_time = 0
                self.last_score = None
                return False

            diff_img = current_small.copy()
            diff_img.difference(self.previous_small)
            self.last_score = diff_img.get_statistics().mean()
            self.previous_small = current_small
            del diff_img

            if self.last_score <= STUCK_MOTION_SCORE_THRESHOLD:
                if self.still_start_time == 0:
                    self.still_start_time = now_ms
                elif time.ticks_diff(now_ms, self.still_start_time) >= STUCK_NO_MOTION_MS:
                    return True
            else:
                self.still_start_time = 0

        except Exception as motion_error:
            # 图像接口不兼容时只关闭挣脱检测，不能让原有比赛主流程退出。
            self.available = False
            self.reset_observation()
            print(">>> STUCK detector disabled:", motion_error)

        return False

    def still_elapsed(self, now_ms):
        if self.still_start_time == 0:
            return 0
        return time.ticks_diff(now_ms, self.still_start_time)

# ===================== globals =====================
# 当前主状态机状态，贯穿 SEARCH/TRACK/GRAB/VERIFY_GRAB/TRANSPORT/UNLOAD/ESCAPE。
ROBOT_STATE = STATE_SEARCH

# 是否已经通过第一次本方普通球的 VERIFY_GRAB，用于目标优先级切换。
# 注意：当前版本尚未实现运输途中在手监测和入区确认，因此它还不等价于“成功入区”。
FIRST_TARGET_DONE = False

# 已通过 VERIFY_GRAB 并开始运输的次数，仅用于流程统计和 debug。
TRANSPORT_COUNT = 0

# 当前夹持目标类型，用于 debug 和卸载后恢复。
CURRENT_HOLDING_TYPE = "NONE"

# SEARCH 选中目标进入 TRACK 后固定来源/类型，避免靠近途中突然切换目标。
TRACK_LOCK_INFO = None

# GRAB/VERIFY_GRAB 使用的完整锁定信息。
LOCKED_TARGET_INFO = None

# TRACK -> GRAB 的 x/y 到位连续计数。
grab_ready_count = 0

# GRAB 阶段夹爪 ROI 连续确认计数。
grab_roi_ready_count = 0

# GRAB 阶段开始闭爪的时间戳，单位 ms。
grab_close_start_time = 0

# TRACK 提前张开夹爪的阶段变量：
# 0=闭爪靠近；正数=开始张开夹爪的时间戳；-1=夹爪已完全张开并继续靠近。
grab_open_start_time = 0

# VERIFY_GRAB 阶段连续确认到锁定小球的帧数。
verify_grab_ready_count = 0

# VERIFY_GRAB 阶段起始时间戳，单位 ms，用于失败超时。
verify_grab_start_time = 0

# UNLOAD 阶段保留计数，当前主要用于状态恢复。
unload_ready_count = 0

# SEARCH -> TRACK 的连续锁定目标计数。
search_target_ready_count = 0

# SEARCH 连续确认中的单个候选，避免三帧内在多个物料之间跳锁。
SEARCH_CANDIDATE_INFO = None

# UART 句柄，由 K230 UART 初始化后赋值。
uart = None

# GRAB/TRANSPORT 近距离兜底/UNLOAD 等定时动作的起始时间戳，单位 ms。
action_timer = 0

# 【新增】SEARCH 致盲冷却计时器
search_cooldown_timer = 0

# TRANSPORT 安全区丢失计数，用于决定停车等待或转向搜索。
transport_lost_count = 0

# TRANSPORT -> UNLOAD 的 x/y 到位连续计数。
transport_xy_ready_count = 0

# 运动检测运行数据集中在该对象中，不再拆分成多个全局计时器和小图变量。
stuck_detector = StuckDetector()


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


# TRANSPORT 阶段发送安全区坐标；angle 字段固定为 0，不再请求下位机自矫正。
def make_transport_xy_payload(cmd, grip_action, safe_x, safe_y):
    return build_fixed5_payload(
        cmd,
        VisionMode.SAFE_XY,
        grip_action,
        safe_x,
        safe_y,
    )


def get_payload_move_cmd(payload):
    # sign = vision_mode * 100 + cmd * 10 + grip_action。
    # 必须从最终 payload 取运动指令，因为 TRACK/TRANSPORT 可能不更新外层 cmd 变量。
    try:
        return (int(payload[0]) % 100) // 10
    except Exception:
        return MoveCmd.STOP


def should_monitor_stuck(robot_state, payload, safe_detected, safe_blob, grip_open_start_time):
    # 上位机通常先于下位机发车按钮上电；第一次普通球抓取成功前不计堵停。
    if not FIRST_TARGET_DONE:
        return False

    move_cmd = get_payload_move_cmd(payload)
    moving = move_cmd in [
        MoveCmd.FORWARD,
        MoveCmd.BACKWARD,
        MoveCmd.TURN_LEFT,
        MoveCmd.TURN_RIGHT,
    ]

    if not moving or robot_state not in [STATE_SEARCH, STATE_TRACK, STATE_TRANSPORT]:
        return False

    # TRACK 已开始开爪后可能停车等待或低速精调，禁止误判为卡住。
    if robot_state == STATE_TRACK and grip_open_start_time != 0:
        return False

    # 已进入安全区目标 Y 前 50 像素区域时，优先使用 V8.5 的卸载超时兜底，不能后退挣脱。
    if robot_state == STATE_TRANSPORT and safe_detected and safe_blob is not None:
        safe_x_ok = abs(safe_blob["cx"] - CENTER_X) <= TRANSPORT_X_DEAD_ZONE
        if safe_x_ok and safe_blob["cy"] >= TRANSPORT_SAFE_TARGET_Y - 50:
            return False

    return True


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


def draw_small_label(img, x, y, text, color):
    # CV 候选可能较多，使用小号 ASCII 字体避免遮挡画面并降低 OSD 绘制负担。
    try:
        img.draw_string(max(0, x), max(0, y), text, color=color, scale=1)
    except Exception:
        pass


def get_display_label(label):
    name_map = {
        "redball": "RED",
        "blueball": "BLUE",
        "blackball": "CORE",
        "yellowball": "DANGER",
        "cv2_core": "CORE",
        "cv2_danger": "DANGER",
        "cv2_raw_verify_core": "CORE_VERIFY",
        "cv2_raw_verify_danger": "DANGER_VERIFY",
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


def is_target_in_grip_roi(target, require_area=True):
    if target is None:
        return False

    x, y, w, h = GRIP_ROI
    tx = target.get("aim_x", target["cx"])
    ty = target.get("aim_y", target["cy"])

    in_x = x <= tx <= x + w
    in_y = y <= ty <= y + h
    area_ok = (not require_area) or target["area"] >= GRIP_BALL_MIN_AREA

    return in_x and in_y and area_ok


def get_cv_grab_point(target):
    if target is None:
        return None, None
    # x 使用检测框中心，y 使用检测框最下沿；该点专门用于 CV 抓取到位判断。
    return target.get("aim_x", target["cx"]), target.get("y2", target["cy"]) - 1


def is_cv_bottom_in_grab_roi(target):
    if target is None or target.get("source") != "CV2":
        return False
    point_x, point_y = get_cv_grab_point(target)
    roi_x, roi_y, roi_w, roi_h = CV_GRAB_ROI
    return (
        roi_x <= point_x <= roi_x + roi_w and
        roi_y <= point_y <= roi_y + roi_h
    )


def is_target_in_verify_grab_roi(target, require_area=True):
    if target is None:
        return False

    x, y, w, h = VERIFY_GRAB_ROI
    tx = target.get("aim_x", target["cx"])
    ty = target.get("aim_y", target["cy"])

    in_x = x <= tx <= x + w
    in_y = y <= ty <= y + h
    area_ok = (not require_area) or target["area"] >= VERIFY_GRAB_BALL_MIN_AREA

    return in_x and in_y and area_ok


def make_cv_verify_evidence_target(verify_info, locked_info):
    rect = verify_info.get("best_rect")
    if rect is None:
        rect = VERIFY_GRAB_ROI
    center_x = verify_info.get("best_x")
    center_y = verify_info.get("best_y")
    if center_x is None:
        center_x = rect[0] + rect[2] // 2
    if center_y is None:
        center_y = rect[1] + rect[3] // 2
    return {
        "source": "CV2_RAW_VERIFY",
        "type": locked_info["type"],
        "label": "cv2_raw_verify_{}".format(locked_info["type"].lower()),
        "color": locked_info.get("color"),
        "verify_key": locked_info.get("verify_key"),
        "rect": rect,
        "cx": center_x,
        "cy": center_y,
        "area": int(verify_info.get("best_area", 0)),
    }


def get_verify_grab_result(fused_targets, cv_verify_info,
                           cv_verify_excess_pixels, locked_info):
    if locked_info is None:
        return None, 0
    locked_source = locked_info.get("source", "YOLO")
    if locked_source == "YOLO":
        # YOLO 球保持严格确认：ROI 内恰好一个球，且类型与锁定目标一致。
        roi_targets = []
        for target in fused_targets:
            if is_target_in_verify_grab_roi(target, require_area=True):
                roi_targets.append(target)
        target_count = len(roi_targets)
        if target_count != 1:
            return None, target_count
        verify_target = roi_targets[0]
        if verify_target.get("type") != locked_info.get("type"):
            return None, target_count
        return verify_target, target_count

    # CV 物料优先用主检测通路的严格框确认；遮挡导致严格框消失时，
    # 才在小型夹爪 ROI 中使用锁定颜色的原始 HSV 证据兜底。
    target_type = locked_info.get("type")
    strict_targets = []
    for target in fused_targets:
        if (
            target.get("source") == "CV2" and
            target.get("type") == target_type and
            is_target_in_verify_grab_roi(target, require_area=False)
        ):
            strict_targets.append(target)
    if strict_targets:
        verify_target = min(
            strict_targets,
            key=lambda target: abs(target.get("aim_x", target["cx"]) - CENTER_X)
        )
        return verify_target, len(strict_targets)

    if target_type not in CV_VERIFY_CONFIG or cv_verify_info is None:
        return None, 0
    config = CV_VERIFY_CONFIG[target_type]
    evidence_ok = (
        cv_verify_info.get("best_rect") is not None and
        cv_verify_info.get("best_area", 0) >= config["min_contour_area"] and
        cv_verify_excess_pixels >= config["min_excess_pixels"]
    )
    if not evidence_ok:
        return None, 0
    return make_cv_verify_evidence_target(cv_verify_info, locked_info), 1


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
                # K230 aidemo.yolov5_det_postprocess 返回格式为：
                # [x, y, width, height]，不是 [x1, y1, x2, y2]。
                # 必须先转换成右下角坐标，否则框在画面靠左上时会被错误缩小。
                append_det(
                    box[0],
                    box[1],
                    box[0] + box[2],
                    box[1] + box[3],
                    score,
                    class_id,
                    None
                )
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
    global transport_lost_count, transport_xy_ready_count

    transport_lost_count = 0
    transport_xy_ready_count = 0


def clamp(value, minimum, maximum):
    value = int(value)
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def check_cv2_apis():
    required = [
        "cvtColor", "inRange", "bitwise_or", "morphologyEx",
        "findContours", "contourArea", "boundingRect", "mean",
        "getStructuringElement",
    ]
    missing = []
    for name in required:
        if not hasattr(cv2, name):
            missing.append(name)
    if missing:
        raise RuntimeError("required cv2 APIs missing: {}".format(missing))


def compile_hsv_ranges():
    compiled = {}
    for target_type in CV_MATERIAL_ORDER:
        pairs = []
        for lower, upper in CV_MATERIAL_CONFIG[target_type]["ranges"]:
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


def evaluate_cv_contour(hsv_img, contour, target_type, config):
    contour_area = float(cv2.contourArea(contour))
    x, y, width, height = cv2.boundingRect(contour)
    x, y, width, height = int(x), int(y), int(width), int(height)
    box_area = max(1, width * height)
    aspect = float(width) / float(max(1, height))
    fill = contour_area / float(box_area)
    metrics = {
        "contour_area_320": contour_area,
        "box_area_320": box_area,
        "aspect": aspect,
        "fill": fill,
        "contrast": 0.0,
        "inner_v": 0.0,
        "outer_v": 0.0,
        "vertices": optional_vertex_count(contour),
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
    return True, "OK", metrics


def map_cv_metrics_to_target(target_type, config, metrics):
    x, y, width, height = metrics["cv_rect"]
    x1 = int(x * SENSOR_WIDTH / CV_WIDTH)
    y1 = int(y * SENSOR_HEIGHT / CV_HEIGHT)
    mapped_width = max(1, int(width * SENSOR_WIDTH / CV_WIDTH))
    mapped_height = max(1, int(height * SENSOR_HEIGHT / CV_HEIGHT))
    x2 = clamp(x1 + mapped_width, 0, SENSOR_WIDTH)
    y2 = clamp(y1 + mapped_height, 0, SENSOR_HEIGHT)
    x1 = clamp(x1, 0, SENSOR_WIDTH - 1)
    y1 = clamp(y1, 0, SENSOR_HEIGHT - 1)
    mapped_width = max(1, x2 - x1)
    mapped_height = max(1, y2 - y1)
    cx = x1 + mapped_width // 2
    cy = y1 + mapped_height // 2
    aim_y = min(y2 - 1, y1 + int(mapped_height * config["aim_y_ratio"]))
    return {
        "source": "CV2",
        "type": target_type,
        "label": "cv2_{}".format(target_type.lower()),
        "color": config["color"],
        "verify_key": config["color"],
        "draw_color": config["draw_color"],
        "rect": (x1, y1, mapped_width, mapped_height),
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "w": mapped_width, "h": mapped_height,
        "cx": cx, "cy": cy, "aim_x": cx, "aim_y": aim_y,
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


def collect_cv_materials(hsv_img, compiled_ranges, open_kernel, close_kernel):
    accepted = []
    rejected = []
    raw_counts = {}
    reject_counts = {}
    for target_type in CV_MATERIAL_ORDER:
        config = CV_MATERIAL_CONFIG[target_type]
        mask = build_mask(hsv_img, compiled_ranges[target_type])
        processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        del mask
        if config["use_close"]:
            closed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, close_kernel)
            del processed
            processed = closed
        contours = get_contours(processed)
        raw_counts[target_type] = len(contours)
        for contour in contours:
            valid, reason, metrics = evaluate_cv_contour(
                hsv_img, contour, target_type, config
            )
            target = map_cv_metrics_to_target(target_type, config, metrics)
            if valid:
                accepted.append(target)
            else:
                key = "{}:{}".format(target_type, reason)
                reject_counts[key] = reject_counts.get(key, 0) + 1
                if metrics["contour_area_320"] >= config["min_contour_area"]:
                    target["drop_reason"] = reason
                    rejected.append(target)
        del contours
        del processed
    rejected.sort(key=lambda item: item["contour_area_320"], reverse=True)
    if len(rejected) > MAX_REJECTED_DRAW:
        rejected = rejected[:MAX_REJECTED_DRAW]
    return accepted, rejected, raw_counts, reject_counts


def boxes_overlap(cv_target, yolo_target):
    margin = YOLO_OVERLAP_MARGIN
    yx1 = clamp(yolo_target["x1"] - margin, 0, SENSOR_WIDTH)
    yy1 = clamp(yolo_target["y1"] - margin, 0, SENSOR_HEIGHT)
    yx2 = clamp(yolo_target["x2"] + margin, 0, SENSOR_WIDTH)
    yy2 = clamp(yolo_target["y2"] + margin, 0, SENSOR_HEIGHT)
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
    accepted, dropped_ball, dropped_safe = [], [], []
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


def select_locked_strict_target(cv_targets, lock_info):
    if lock_info is None:
        return None
    last_x = lock_info.get("cx")
    last_y = lock_info.get("cy")
    best = None
    best_distance_sq = None
    max_distance_sq = STRICT_MATCH_MAX_DISTANCE * STRICT_MATCH_MAX_DISTANCE
    for target in cv_targets:
        if target["type"] != lock_info.get("type"):
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
        if distance_sq > max_distance_sq:
            continue
        if best is None or distance_sq < best_distance_sq:
            best = target
            best_distance_sq = distance_sq
    return best


def display_rect_to_cv(rect):
    x, y, width, height = rect
    cv_x = clamp(int(x * CV_WIDTH / SENSOR_WIDTH), 0, CV_WIDTH - 1)
    cv_y = clamp(int(y * CV_HEIGHT / SENSOR_HEIGHT), 0, CV_HEIGHT - 1)
    cv_x2 = clamp(int((x + width) * CV_WIDTH / SENSOR_WIDTH), cv_x + 1, CV_WIDTH)
    cv_y2 = clamp(int((y + height) * CV_HEIGHT / SENSOR_HEIGHT), cv_y + 1, CV_HEIGHT)
    return cv_x, cv_y, cv_x2 - cv_x, cv_y2 - cv_y


def cv_rect_to_display(x, y, width, height):
    return (
        int(x * SENSOR_WIDTH / CV_WIDTH),
        int(y * SENSOR_HEIGHT / CV_HEIGHT),
        max(1, int(width * SENSOR_WIDTH / CV_WIDTH)),
        max(1, int(height * SENSOR_HEIGHT / CV_HEIGHT)),
    )


def mask_pixel_count(mask):
    if mask is None or len(mask.shape) < 2:
        return 0
    height = int(mask.shape[0])
    width = int(mask.shape[1])
    if width <= 0 or height <= 0:
        return 0
    return int(float(cv2.mean(mask)[0]) * width * height / 255.0 + 0.5)


def measure_cv_roi_raw_pixels(hsv_img, target_type, compiled_ranges, roi_rect):
    cv_x, cv_y, cv_width, cv_height = display_rect_to_cv(roi_rect)
    full_mask = build_mask(hsv_img, compiled_ranges[target_type])
    roi_mask = full_mask[cv_y:cv_y + cv_height, cv_x:cv_x + cv_width]
    pixels = mask_pixel_count(roi_mask)
    del roi_mask
    del full_mask
    return pixels


def collect_cv_roi_evidence(hsv_img, locked_type, compiled_ranges,
                            open_kernel, close_kernel, roi_rect):
    result = {
        "raw_pixels": 0, "contours": 0, "best_area": 0.0,
        "best_rect": None, "best_x": None,
        "best_y": None, "best_aim_y": None,
    }
    if locked_type not in CV_VERIFY_CONFIG:
        return result
    cv_x, cv_y, cv_width, cv_height = display_rect_to_cv(roi_rect)
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
    for contour in contours:
        area = float(cv2.contourArea(contour))
        local_x, local_y, width, height = cv2.boundingRect(contour)
        global_x = cv_x + int(local_x)
        global_y = cv_y + int(local_y)
        draw_rect = cv_rect_to_display(global_x, global_y, int(width), int(height))
        center_x = draw_rect[0] + draw_rect[2] // 2
        center_y = draw_rect[1] + draw_rect[3] // 2
        aim_y = draw_rect[1] + draw_rect[3] - 1
        score = area
        if best_score is None or score > best_score:
            best_score = score
            result["best_area"] = area
            result["best_rect"] = draw_rect
            result["best_x"] = center_x
            result["best_y"] = center_y
            result["best_aim_y"] = aim_y
    del contours
    del processed
    del roi_mask
    del full_mask
    return result


def get_yolo_ball_targets(detections):
    targets = []
    type_map = {
        "redball": "NORMAL_RED",
        "blueball": "NORMAL_BLUE",
    }
    color_map = {
        "redball": "RED",
        "blueball": "BLUE",
    }
    for det in detections:
        if det["label"] not in type_map:
            continue
        if det["area"] < MIN_DRAW_AREA or det["area"] > MAX_DRAW_AREA:
            continue
        targets.append({
            "cx": det["cx"],
            "cy": det["cy"],
            "aim_x": det["cx"],
            "aim_y": det["cy"],
            "area": det["area"],
            "type": type_map[det["label"]],
            "rect": det["rect"],
            "label": det["label"],
            "source": "YOLO",
            "color": color_map[det["label"]],
            "verify_key": color_map[det["label"]],
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
    valid_targets = []
    own_type = "NORMAL_RED" if TEAM_COLOR == "RED" else "NORMAL_BLUE"
    filter_labels = ["blueboard", "redboard"]
    for t in targets_list:
        if t["type"] != own_type:
            continue
        in_safe_zone = False
        tx, ty = t["cx"], t["cy"]
        for det in detections:
            if det["label"] in filter_labels:
                pad = 10
                if (det["x1"] - pad) <= tx <= (det["x2"] + pad) and (det["y1"] - pad) <= ty <= (det["y2"] + pad):
                    in_safe_zone = True
                    break
        if not in_safe_zone:
            valid_targets.append(t)
    if not valid_targets:
        return None
    def score_target(t):
        center_penalty = abs(t["cx"] - CENTER_X) * CENTER_SCORE_PENALTY
        base_score = t["area"] - center_penalty
        if last_tracked_cx is not None and last_tracked_cy is not None:
            if t["type"] == last_tracked_type:
                dist = abs(t["cx"] - last_tracked_cx) + abs(t["cy"] - last_tracked_cy)
                if dist < TRACK_MATCH_DIST:
                    base_score += TRACK_INERTIA_BONUS
        return base_score
    return max(valid_targets, key=score_target)


def get_cv_priority_target(cv_targets):
    # 决赛第一件物料必须先完成本方普通球；之后 CORE 硬优先于 DANGER。
    if not FIRST_TARGET_DONE:
        return None
    for target_type in CV_MATERIAL_ORDER:
        candidates = [target for target in cv_targets if target["type"] == target_type]
        if candidates:
            return max(
                candidates,
                key=lambda target: (
                    1 if target.get("yolo_support", False) else 0,
                    target["area"] - abs(target["cx"] - CENTER_X) * 20,
                ),
            )
    return None


def get_locked_track_target(yolo_targets, cv_targets, lock_info):
    if lock_info is None:
        return None
    if lock_info.get("source") == "CV2":
        return select_locked_strict_target(cv_targets, lock_info)
    source_pool = yolo_targets
    candidates = []
    for target in source_pool:
        if target["type"] == lock_info.get("type"):
            candidates.append(target)
    if not candidates:
        return None

    ref_x = lock_info.get("cx", CENTER_X)
    ref_y = lock_info.get("cy", SENSOR_HEIGHT // 2)
    return min(
        candidates,
        key=lambda target: abs(target["cx"] - ref_x) + abs(target["cy"] - ref_y)
    )


def make_target_lock_info(target):
    if target is None:
        return None
    return {
        "source": target.get("source", "YOLO"),
        "type": target["type"],
        "color": target.get("color"),
        "verify_key": target.get("verify_key"),
        "cx": target.get("aim_x", target["cx"]),
        "cy": target.get("aim_y", target["cy"]),
    }


def is_first_allowed_target_info(locked_info):
    if locked_info is None:
        return False
    own_type = "NORMAL_RED" if TEAM_COLOR == "RED" else "NORMAL_BLUE"
    return locked_info.get("source") == "YOLO" and locked_info.get("type") == own_type


def draw_cv_detections(img, accepted, rejected, dropped_ball, dropped_safe):
    if SHOW_REJECTED_CV:
        for target in rejected:
            img.draw_rectangle(target["rect"], color=(255, 128, 0), thickness=1)
            draw_small_label(img, target["x1"], target["y1"] + target["h"],
                             "REJ:{}".format(target["drop_reason"]), (255, 128, 0))
    for target in dropped_ball:
        img.draw_rectangle(target["rect"], color=(100, 100, 100), thickness=1)
        draw_small_label(img, target["x1"], max(0, target["y1"] - 12), "DROP_B", (100, 100, 100))

    for target in dropped_safe:
        img.draw_rectangle(target["rect"], color=(255, 128, 0), thickness=1)
        draw_small_label(img, target["x1"], max(0, target["y1"] - 12), "DROP_S", (255, 128, 0))

    for target in accepted:
        color = target["draw_color"]
        img.draw_rectangle(target["rect"], color=color, thickness=3)
        img.draw_cross(target["aim_x"], target["aim_y"], color=color, size=10)
        draw_small_label(
            img,
            target["x1"],
            max(0, target["y1"] - 12),
            "CV2:{} C:{:.1f}".format(target["type"], target["contrast"]),
            color,
        )


try:
    # ===================== 初始化阶段 =====================
    sensor = Sensor(id=sensor_id)
    sensor.reset()

    sensor.set_framesize(width=SENSOR_WIDTH, height=SENSOR_HEIGHT, chn=DISPLAY_CHN)
    sensor.set_pixformat(Sensor.YUV420SP, chn=DISPLAY_CHN)

    sensor.set_framesize(width=CV_WIDTH, height=CV_HEIGHT, chn=CV_CHN)
    sensor.set_pixformat(Sensor.RGB888, chn=CV_CHN)

    sensor.set_framesize(width=SENSOR_WIDTH, height=SENSOR_HEIGHT, chn=YOLO_CHN)
    sensor.set_pixformat(Sensor.RGBP888, chn=YOLO_CHN)

    bind_info = sensor.bind_info(x=0, y=0, chn=DISPLAY_CHN)
    Display.bind_layer(**bind_info, layer=Display.LAYER_VIDEO1)
    Display.init(
        Display.ST7701,
        width=SENSOR_WIDTH,
        height=SENSOR_HEIGHT,
        to_ide=True,
        osd_num=1,
    )
    MediaManager.init()
    sensor.run()
    osd_img = image.Image(SENSOR_WIDTH, SENSOR_HEIGHT, image.ARGB8888)

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

    check_cv2_apis()
    compiled_ranges = compile_hsv_ranges()
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    UART_init()
    uart = UART(
        UART.UART2,
        baudrate=115200,
        bits=UART.EIGHTBITS,
        parity=UART.PARITY_NONE,
        stop=UART.STOPBITS_ONE,
    )

    # 默认闭合夹爪，避免搜索/靠近过程中张开的夹爪误兜其他球。
    send_payload_if_needed(make_normal_payload(MoveCmd.STOP, GripAction.GRAB))
    print("system start... ({})".format(VERSION))
    print(">>> YOLO flow: SEARCH -> TRACK -> GRAB -> VERIFY_GRAB -> TRANSPORT")
    print(">>> CV flow: SEARCH -> TRACK -> early GRAB -> direct TRANSPORT")
    print(">>> channels: CH0 DISPLAY, CH1 CV2 RGB888/HWC, CH2 YOLO RGBP888")
    print(">>> priority: own YOLO ball > CV2 CORE > CV2 DANGER")
    print(">>> CV grab trigger: box-bottom center enters independent CV_GRAB_ROI for 3 frames")

    action_timer = time.ticks_ms()
    STATE_NAMES = {
        STATE_SEARCH: "SEARCH",
        STATE_TRACK: "TRACK",
        STATE_GRAB: "GRAB",
        STATE_VERIFY_GRAB: "VERIFY_GRAB",
        STATE_TRANSPORT: "TRANSPORT",
        STATE_UNLOAD: "UNLOAD",
        STATE_ESCAPE: "ESCAPE",
    }
    LAST_ROBOT_STATE = -1
    debug_print_count = 0
    frame_count = 0
    last_detections = []
    last_has_own_ball = False
    empty_verify_baseline_sums = {"CORE": 0, "DANGER": 0}
    empty_verify_baseline_pixels = {"CORE": None, "DANGER": None}
    empty_verify_baseline_count = 0
    first_cv_shape_printed = False
    last_cv_candidates = []
    last_rejected_cv = []
    last_cv_raw_counts = {"CORE": 0, "DANGER": 0}
    last_cv_reject_counts = {}

    while True:
        # ===================== 主循环 =====================
        os.exitpoint()
        frame_count += 1

        baseline_ready = empty_verify_baseline_count >= EMPTY_BASELINE_CALIBRATION_FRAMES
        cv_state_locked = (
            ROBOT_STATE in [STATE_TRACK, STATE_GRAB, STATE_VERIFY_GRAB] and
            ((TRACK_LOCK_INFO is not None and TRACK_LOCK_INFO.get("source") == "CV2") or
             (LOCKED_TARGET_INFO is not None and LOCKED_TARGET_INFO.get("source") == "CV2"))
        )
        if not baseline_ready:
            run_yolo = True
            run_cv = True
            scheduler_mode = "CALIB_BOTH"
        elif cv_state_locked:
            run_yolo = frame_count % YOLO_EVERY_N_FRAMES_WHEN_CV_LOCKED == 0
            run_cv = True
            scheduler_mode = "CV_LOCK_ALT"
        elif ROBOT_STATE == STATE_TRANSPORT:
            run_yolo = True
            run_cv = False
            scheduler_mode = "TRANSPORT_YOLO"
        elif ROBOT_STATE in [STATE_GRAB, STATE_VERIFY_GRAB]:
            # 抓取 YOLO 球的短暂阶段同时开启 CV，供“ROI 内恰好一个物料”复核。
            run_yolo = True
            run_cv = True
            scheduler_mode = "YOLO_GRAB_BOTH"
        else:
            run_yolo = True
            run_cv = ROBOT_STATE == STATE_SEARCH and not last_has_own_ball
            scheduler_mode = "SEARCH_BOTH" if run_cv else "YOLO_ONLY"

        # CV 运算可跳过；第一次抓球后仍保留 CH1 image.Image 给堵停检测。
        need_cv_snapshot = run_cv or FIRST_TARGET_DONE
        yolo_img = sensor.snapshot(chn=YOLO_CHN) if run_yolo else None
        cv_img = sensor.snapshot(chn=CV_CHN) if need_cv_snapshot else None
        yolo_input = None
        current_hsv_img = None
        try:
            osd_img.clear()
        except Exception:
            pass

        if run_yolo:
            yolo_input = yolo_img.to_numpy_ref()
            res = yolo.run(yolo_input)
            dets = parse_yolo_results(res)
            last_detections = dets
            if debug_print_count < 3:
                print("YOLO raw result:", res)
        else:
            dets = last_detections
        debug_print_count += 1
        if ROBOT_STATE != LAST_ROBOT_STATE:
            print("\n>>> [STATE] ---> {}".format(STATE_NAMES.get(ROBOT_STATE, "UNKNOWN")))
            LAST_ROBOT_STATE = ROBOT_STATE

        safe_blob = detect_safe_zone_from_yolo(dets)
        safe_detected = safe_blob is not None
        yolo_targets = get_yolo_ball_targets(dets)
        yolo_normal_dets = []
        yolo_safe_dets = []
        yolo_assist_dets = []
        for det in dets:
            if det["label"] in YOLO_NORMAL_BALL_LABELS:
                yolo_normal_dets.append(det)
            elif det["label"] in YOLO_SAFE_LABELS:
                yolo_safe_dets.append(det)
            elif det["label"] in YOLO_ASSIST_LABELS:
                yolo_assist_dets.append(det)

        own_label = "redball" if TEAM_COLOR == "RED" else "blueball"
        last_has_own_ball = False
        for det in yolo_normal_dets:
            if det["label"] == own_label:
                last_has_own_ball = True
                break

        if run_cv:
            small_rgb = cv_img.to_numpy_ref()
            if not first_cv_shape_printed:
                print("CV RGB888 HWC shape:", small_rgb.shape)
                if len(small_rgb.shape) != 3 or small_rgb.shape[2] != 3:
                    raise RuntimeError("CH1 RGB888 expected HWC, got {}".format(small_rgb.shape))
                first_cv_shape_printed = True
            current_hsv_img = cv2.cvtColor(small_rgb, cv2.COLOR_RGB2HSV)
            (
                last_cv_candidates,
                last_rejected_cv,
                last_cv_raw_counts,
                last_cv_reject_counts,
            ) = collect_cv_materials(
                current_hsv_img, compiled_ranges, open_kernel, close_kernel
            )
            if empty_verify_baseline_count < EMPTY_BASELINE_CALIBRATION_FRAMES:
                for baseline_type in CV_MATERIAL_ORDER:
                    empty_verify_baseline_sums[baseline_type] += measure_cv_roi_raw_pixels(
                        current_hsv_img, baseline_type, compiled_ranges, VERIFY_GRAB_ROI
                    )
                empty_verify_baseline_count += 1
                if empty_verify_baseline_count == EMPTY_BASELINE_CALIBRATION_FRAMES:
                    for baseline_type in CV_MATERIAL_ORDER:
                        empty_verify_baseline_pixels[baseline_type] = int(
                            empty_verify_baseline_sums[baseline_type] /
                            EMPTY_BASELINE_CALIBRATION_FRAMES
                        )
                    print(">>> EMPTY_BASELINES READY CORE={} DANGER={}".format(
                        empty_verify_baseline_pixels["CORE"],
                        empty_verify_baseline_pixels["DANGER"]
                    ))
            del small_rgb
        else:
            last_cv_candidates = []
            last_rejected_cv = []
            last_cv_raw_counts = {"CORE": 0, "DANGER": 0}
            last_cv_reject_counts = {}

        cv_targets, cv_dropped_ball, cv_dropped_safe = fuse_cv_candidates(
            last_cv_candidates, yolo_normal_dets, yolo_safe_dets, yolo_assist_dets
        )

        if ROBOT_STATE == STATE_SEARCH:
            # 本方 YOLO 球硬优先；看不到本方球且已经完成首球后才选择 CV 物料。
            control_target = get_priority_target(yolo_targets, dets)
            if control_target is None and baseline_ready:
                control_target = get_cv_priority_target(cv_targets)
        elif ROBOT_STATE == STATE_TRACK or ROBOT_STATE == STATE_GRAB:
            control_target = get_locked_track_target(yolo_targets, cv_targets, TRACK_LOCK_INFO)
        else:
            control_target = None

        # CV 和 YOLO 在 SEARCH/TRACK/GRAB 中统一使用主检测框坐标。
        # 只有闭爪后的 VERIFY_GRAB，才在小型夹爪 ROI 中计算原始颜色证据。
        cv_verify_info = None
        cv_verify_excess = 0
        if (
            ROBOT_STATE == STATE_VERIFY_GRAB and
            LOCKED_TARGET_INFO is not None and
            LOCKED_TARGET_INFO.get("source") == "CV2" and
            current_hsv_img is not None
        ):
            target_type = LOCKED_TARGET_INFO["type"]
            cv_verify_info = collect_cv_roi_evidence(
                current_hsv_img, target_type, compiled_ranges,
                open_kernel, close_kernel, VERIFY_GRAB_ROI
            )
            baseline_pixels = empty_verify_baseline_pixels.get(target_type)
            if baseline_pixels is None:
                baseline_pixels = cv_verify_info["raw_pixels"]
            cv_verify_excess = max(
                0, cv_verify_info["raw_pixels"] - baseline_pixels
            )

        sections = get_sections_from_yolo(dets)

        # ==========================================
        # 更新追踪记忆
        if control_target is not None:
            last_tracked_cx = control_target["cx"]
            last_tracked_cy = control_target["cy"]
            last_tracked_type = control_target["type"]
            if (ROBOT_STATE == STATE_TRACK or ROBOT_STATE == STATE_GRAB) and TRACK_LOCK_INFO is not None:
                TRACK_LOCK_INFO["cx"] = control_target.get("aim_x", control_target["cx"])
                TRACK_LOCK_INFO["cy"] = control_target.get("aim_y", control_target["cy"])
        elif ROBOT_STATE == STATE_SEARCH:
            last_tracked_cx = None
            last_tracked_cy = None
            last_tracked_type = None
        # ==========================================

        draw_cv_detections(osd_img, cv_targets, last_rejected_cv,
                           cv_dropped_ball, cv_dropped_safe)
        draw_yolo_detections(osd_img, dets, control_target, safe_blob)
        if debug_print_count <= 3:
            print(
                "YOLO targets:", len(yolo_targets),
                "safe:", safe_detected,
                "CV ok/drop_ball/drop_safe:",
                len(cv_targets), len(cv_dropped_ball), len(cv_dropped_safe),
                "verify baseline:", empty_verify_baseline_pixels,
                "sections:", len(sections)
            )

        cmd = MoveCmd.STOP
        grip = GripAction.RELEASE
        payload = make_normal_payload(cmd, grip)
        transport_debug_safe_x = None
        transport_debug_safe_y = None
        transport_debug_box_area = None
        transport_debug_xy_ready_count = transport_xy_ready_count
        verify_grab_debug_detected = False
        verify_grab_debug_elapsed = 0
        verify_grab_debug_target_count = 0

        if ROBOT_STATE == STATE_SEARCH:
            # 搜索和靠近阶段默认闭爪，避免张开的夹爪误抓/误兜非目标球。
            grip = GripAction.GRAB
            TRACK_LOCK_INFO = None
            LOCKED_TARGET_INFO = None
            grab_ready_count = 0
            grab_roi_ready_count = 0
            grab_close_start_time = 0
            grab_open_start_time = 0
            verify_grab_ready_count = 0
            verify_grab_start_time = 0
            unload_ready_count = 0
            reset_transport_controller()

            is_cooldown = time.ticks_diff(time.ticks_ms(), search_cooldown_timer) < SEARCH_COOLDOWN_MS

            if control_target and control_target["area"] >= SEARCH_LOCK_MIN_AREA and not is_cooldown:
                candidate_x = control_target.get("aim_x", control_target["cx"])
                candidate_y = control_target.get("aim_y", control_target["cy"])
                same_candidate = (
                    SEARCH_CANDIDATE_INFO is not None and
                    SEARCH_CANDIDATE_INFO.get("source") == control_target.get("source") and
                    SEARCH_CANDIDATE_INFO.get("type") == control_target.get("type") and
                    abs(SEARCH_CANDIDATE_INFO.get("cx", candidate_x) - candidate_x) +
                    abs(SEARCH_CANDIDATE_INFO.get("cy", candidate_y) - candidate_y) <=
                    TRACK_MATCH_DIST
                )
                if same_candidate:
                    search_target_ready_count += 1
                else:
                    search_target_ready_count = 1
                SEARCH_CANDIDATE_INFO = {
                    "source": control_target.get("source"),
                    "type": control_target.get("type"),
                    "cx": candidate_x,
                    "cy": candidate_y,
                }
            else:
                search_target_ready_count = 0
                SEARCH_CANDIDATE_INFO = None

            if search_target_ready_count >= SEARCH_ENTER_TRACK_FRAMES:
                ROBOT_STATE = STATE_TRACK
                TRACK_LOCK_INFO = make_target_lock_info(control_target)
                SEARCH_CANDIDATE_INFO = None
                cmd = MoveCmd.STOP
                payload = make_normal_payload(cmd, grip)
                print(">>> enter TRACK: type={} cx={} cy={} area={}".format(
                    control_target["type"], control_target["cx"], control_target["cy"], control_target["area"]
                ))
            else:
                cmd = MoveCmd.TURN_LEFT
                payload = make_normal_payload(cmd, grip)

        elif ROBOT_STATE == STATE_TRACK:
            now_ms = time.ticks_ms()

            # grab_open_start_time 的含义：0=闭爪靠近，正数=停车开爪等待，-1=已开爪靠近。
            if grab_open_start_time == 0:
                grip = GripAction.GRAB
            else:
                grip = GripAction.RELEASE

            if control_target:
                transport_lost_count = 0

                target_aim_x = control_target.get("aim_x", control_target["cx"])
                target_aim_y = control_target.get("aim_y", control_target["cy"])
                open_x_ok = abs(target_aim_x - CENTER_X) <= TRACK_X_DEAD_ZONE
                open_y_ok = target_aim_y >= TRACK_GRIP_OPEN_TARGET_Y

                if grab_open_start_time == 0:
                    # 第一阶段：闭爪执行 BALL_XY。x 对准且达到开爪阈值连续 3 帧后停车开爪。
                    if open_x_ok and open_y_ok:
                        grab_ready_count += 1
                    else:
                        grab_ready_count = 0

                    if grab_ready_count >= 3:
                        grab_open_start_time = now_ms
                        grab_ready_count = 0
                        cmd = MoveCmd.STOP
                        grip = GripAction.RELEASE
                        payload = make_normal_payload(cmd, grip)
                        print(">>> TRACK open gripper: type={} cx={} cy={}".format(
                            control_target["type"], control_target["cx"], control_target["cy"]
                        ))
                    else:
                        payload = make_track_payload(
                            grip_action=GripAction.GRAB,
                            target_x=target_aim_x,
                            target_y=target_aim_y,
                        )

                elif grab_open_start_time > 0:
                    # 第二阶段：停车开爪，直接在流程内固定等待 500 ms。
                    grab_ready_count = 0
                    cmd = MoveCmd.STOP
                    grip = GripAction.RELEASE

                    if time.ticks_diff(now_ms, grab_open_start_time) >= 500:
                        grab_open_start_time = -1
                        print(">>> TRACK gripper opened: resume BALL_XY")

                    payload = make_normal_payload(cmd, grip)

                else:
                    # 第三阶段：夹爪已经张开，继续 BALL_XY 靠近目标。
                    # YOLO 球仍等待中心点达到 y=420；CV 物料改用框底进入 CV_GRAB_ROI，
                    # 避免高物料的中心点尚未到 420 就先触底并被 BORDER 过滤。
                    grip = GripAction.RELEASE
                    is_cv_target = control_target.get("source") == "CV2"
                    if is_cv_target:
                        # CV 使用独立 ROI，框底中心点必须真正进入该矩形。
                        target_ready = is_cv_bottom_in_grab_roi(control_target)
                        required_grab_frames = 3
                    else:
                        # YOLO 保持原有中心点 x/y 到位逻辑，不受 CV_GRAB_ROI 影响。
                        target_ready = (
                            abs(target_aim_x - CENTER_X) <= TRACK_X_DEAD_ZONE and
                            target_aim_y >= TRACK_GRAB_TARGET_Y
                        )
                        required_grab_frames = GRAB_CONFIRM_FRAMES

                    if target_ready:
                        grab_ready_count += 1
                    else:
                        grab_ready_count = 0

                    if grab_ready_count >= required_grab_frames:
                        ROBOT_STATE = STATE_GRAB
                        LOCKED_TARGET_INFO = make_target_lock_info(control_target)
                        search_target_ready_count = 0
                        grab_roi_ready_count = 0
                        grab_open_start_time = 0
                        verify_grab_ready_count = 0
                        verify_grab_start_time = 0
                        action_timer = now_ms
                        cmd = MoveCmd.STOP
                        if is_cv_target:
                            # CV 目标在严格框仍可见时立即闭爪；之后不再依赖它继续可见。
                            grip = GripAction.GRAB
                            grab_close_start_time = now_ms
                            print(">>> CV enter GRAB and close now: type={} x={} box_y2={}".format(
                                LOCKED_TARGET_INFO["type"], target_aim_x,
                                control_target.get("y2", target_aim_y)
                            ))
                        else:
                            grab_close_start_time = 0
                            print(">>> enter GRAB with gripper opened: type={} cx={} cy={} area={}".format(
                                LOCKED_TARGET_INFO["type"], control_target["cx"],
                                control_target["cy"], control_target["area"]
                            ))
                        payload = make_normal_payload(cmd, grip)
                    else:
                        payload = make_track_payload(
                            grip_action=grip,
                            target_x=target_aim_x,
                            target_y=target_aim_y,
                        )
            else:
                grab_ready_count = 0
                transport_lost_count += 1
                if transport_lost_count <= TRANSPORT_LOST_MAX:
                    cmd = MoveCmd.STOP
                    payload = make_normal_payload(cmd, grip)
                else:
                    was_gripper_open = grab_open_start_time != 0
                    ROBOT_STATE = STATE_SEARCH
                    TRACK_LOCK_INFO = None
                    LOCKED_TARGET_INFO = None
                    transport_lost_count = 0
                    search_target_ready_count = 0
                    grab_open_start_time = 0
                    if was_gripper_open:
                        # 已开爪时先停车，下一帧 SEARCH 会闭爪后再开始转向。
                        cmd = MoveCmd.STOP
                        grip = GripAction.RELEASE
                    else:
                        cmd = MoveCmd.TURN_LEFT
                        grip = GripAction.GRAB
                    payload = make_normal_payload(cmd, grip)

        elif ROBOT_STATE == STATE_GRAB:
            now_ms = time.ticks_ms()

            if grab_close_start_time != 0:
                cmd = MoveCmd.STOP
                grip = GripAction.GRAB

                if time.ticks_diff(now_ms, grab_close_start_time) >= GRIP_CLOSE_TIME_MS:
                    if (
                        LOCKED_TARGET_INFO is not None and
                        LOCKED_TARGET_INFO.get("source") == "CV2"
                    ):
                        # CV 黑方块/黄锥进入抓取区后已立即闭爪。
                        # 高物料被夹爪或画面底边遮挡是正常现象，不再执行视觉抓取复核。
                        TRANSPORT_COUNT += 1
                        CURRENT_HOLDING_TYPE = LOCKED_TARGET_INFO["type"]
                        ROBOT_STATE = STATE_TRANSPORT
                        TRACK_LOCK_INFO = None
                        search_target_ready_count = 0
                        grab_ready_count = 0
                        grab_roi_ready_count = 0
                        grab_close_start_time = 0
                        grab_open_start_time = 0
                        verify_grab_ready_count = 0
                        verify_grab_start_time = 0
                        unload_ready_count = 0
                        reset_transport_controller()
                        action_timer = 0
                        print(">>> CV GRAB closed: skip VERIFY_GRAB, holding={}, transport_count={}".format(
                            CURRENT_HOLDING_TYPE, TRANSPORT_COUNT
                        ))
                    else:
                        ROBOT_STATE = STATE_VERIFY_GRAB
                        verify_grab_ready_count = 0
                        verify_grab_start_time = now_ms
                        grab_close_start_time = 0
                        grab_open_start_time = 0
                        print(">>> enter VERIFY_GRAB: locked={}".format(LOCKED_TARGET_INFO))

                payload = make_normal_payload(cmd, grip)

            elif control_target:
                grip = GripAction.RELEASE

                require_grip_area = control_target.get("source") == "YOLO"
                if is_target_in_grip_roi(control_target, require_area=require_grip_area):
                    grab_roi_ready_count += 1
                else:
                    grab_roi_ready_count = 0

                if grab_roi_ready_count < GRIP_BALL_CONFIRM_FRAMES:
                    payload = make_track_payload(
                        grip_action=grip,
                        target_x=control_target.get("aim_x", control_target["cx"]),
                        target_y=control_target.get("aim_y", control_target["cy"]),
                    )
                else:
                    cmd = MoveCmd.STOP
                    grip = GripAction.GRAB
                    grab_close_start_time = now_ms
                    payload = make_normal_payload(cmd, grip)

            else:
                grab_roi_ready_count = 0
                grab_close_start_time = 0
                grab_open_start_time = 0
                ROBOT_STATE = STATE_SEARCH
                TRACK_LOCK_INFO = None
                LOCKED_TARGET_INFO = None
                # 夹爪张开阶段丢失目标时先原地停车并保持张开，
                # 下一帧进入 SEARCH 后再闭爪搜索，避免张爪转向时误兜其他球。
                cmd = MoveCmd.STOP
                grip = GripAction.RELEASE
                payload = make_normal_payload(cmd, grip)
                print(">>> GRAB failed: target lost, back to SEARCH")

        elif ROBOT_STATE == STATE_VERIFY_GRAB:
            now_ms = time.ticks_ms()
            cmd = MoveCmd.STOP
            grip = GripAction.GRAB

            verify_target, verify_grab_debug_target_count = get_verify_grab_result(
                yolo_targets + cv_targets,
                cv_verify_info, cv_verify_excess, LOCKED_TARGET_INFO
            )
            verify_grab_debug_detected = verify_target is not None
            verify_grab_debug_elapsed = time.ticks_diff(now_ms, verify_grab_start_time)

            if verify_target is not None:
                verify_grab_ready_count += 1
            else:
                verify_grab_ready_count = 0

            if verify_grab_ready_count >= VERIFY_GRAB_CONFIRM_FRAMES:
                TRANSPORT_COUNT += 1
                CURRENT_HOLDING_TYPE = (
                    LOCKED_TARGET_INFO["type"] if LOCKED_TARGET_INFO is not None
                    else verify_target["type"]
                )
                if (not FIRST_TARGET_DONE) and is_first_allowed_target_info(LOCKED_TARGET_INFO):
                    FIRST_TARGET_DONE = True

                ROBOT_STATE = STATE_TRANSPORT
                TRACK_LOCK_INFO = None
                search_target_ready_count = 0
                grab_ready_count = 0
                grab_roi_ready_count = 0
                grab_close_start_time = 0
                grab_open_start_time = 0
                verify_grab_ready_count = 0
                verify_grab_start_time = 0
                unload_ready_count = 0
                reset_transport_controller()
                # TRANSPORT 中复用 action_timer；0 表示尚未进入目标 Y 前 50 像素区域。
                action_timer = 0
                print(">>> VERIFY_GRAB success: holding={}, transport_count={}".format(
                    CURRENT_HOLDING_TYPE, TRANSPORT_COUNT
                ))

            elif verify_grab_debug_elapsed >= VERIFY_GRAB_TIMEOUT_MS:
                ROBOT_STATE = STATE_SEARCH
                CURRENT_HOLDING_TYPE = "NONE"
                TRACK_LOCK_INFO = None
                LOCKED_TARGET_INFO = None
                search_target_ready_count = 0
                grab_ready_count = 0
                grab_roi_ready_count = 0
                grab_close_start_time = 0
                grab_open_start_time = 0
                verify_grab_ready_count = 0
                verify_grab_start_time = 0
                unload_ready_count = 0
                reset_transport_controller()
                cmd = MoveCmd.STOP
                grip = GripAction.RELEASE
                print(">>> VERIFY_GRAB failed: timeout, release and back to SEARCH")

            payload = make_normal_payload(cmd, grip)

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

                # 持续发送 SAFE_XY 坐标让下位机靠近安全区，angle 字段固定为 0。
                payload = make_transport_xy_payload(
                    MoveCmd.FORWARD,
                    GripAction.GRAB,
                    safe_x,
                    safe_y,
                )

                if safe_x_ok and safe_y_ok:
                    transport_xy_ready_count += 1
                else:
                    transport_xy_ready_count = 0

                # 进入目标 Y 前 50 像素区域且横向已对准后开始计时。
                # 50 像素和 3000 ms 直接写在判断中，不增加新的全局配置变量。
                if safe_x_ok and safe_y >= TRANSPORT_SAFE_TARGET_Y - 50:##################
                    if action_timer == 0:
                        action_timer = now_ms
                        print(">>> TRANSPORT near zone: start 3000 ms fallback timer")
                else:
                    action_timer = 0

                transport_debug_xy_ready_count = transport_xy_ready_count

                print("TRANSPORT XY safe_x={} safe_y={} err={} box={} x_ok={} y_ok={} xy_cnt={}".format(
                    safe_x, safe_y, err_x, safe_box_area, safe_x_ok, safe_y_ok,
                    transport_debug_xy_ready_count
                ))

                if transport_xy_ready_count >= TRANSPORT_XY_CONFIRM_FRAMES:
                    # 不再进入 SELF_ALIGN；到位后先停车，再直接进入 UNLOAD。
                    payload = make_transport_xy_payload(
                        MoveCmd.STOP,
                        GripAction.GRAB,
                        safe_x,
                        safe_y,
                    )
                    ROBOT_STATE = STATE_UNLOAD
                    action_timer = now_ms
                    reset_transport_controller()
                    print(">>> TRANSPORT XY ready: direct to UNLOAD")
                elif action_timer != 0 and time.ticks_diff(now_ms, action_timer) >= 3000:#############
                    # 已进入近距离区域 3 秒但仍未达到目标 Y，认为小车可能抵住围栏。
                    payload = make_transport_xy_payload(
                        MoveCmd.STOP,
                        GripAction.GRAB,
                        safe_x,
                        safe_y,
                    )
                    ROBOT_STATE = STATE_UNLOAD
                    action_timer = now_ms
                    reset_transport_controller()
                    print(">>> TRANSPORT near timeout: fallback to UNLOAD")
            else:
                unload_ready_count = 0
                transport_xy_ready_count = 0
                action_timer = 0
                transport_lost_count += 1

                if transport_lost_count >= TRANSPORT_LOST_MAX:
                    cmd = TRANSPORT_LOST_SEARCH_CMD
                else:
                    cmd = MoveCmd.STOP

                payload = make_normal_payload(cmd, grip)

                print("TRANSPORT safe_detected=0 lost_cnt={} cmd={}".format(
                    transport_lost_count, cmd
                ))

        elif ROBOT_STATE == STATE_ESCAPE:
            # ESCAPE 全程使用普通运动模式并保持闭爪，避免视觉 PID 干扰和运输途中掉球。
            elapsed = time.ticks_diff(time.ticks_ms(), action_timer)
            grip = GripAction.GRAB

            if elapsed < 200:
                # 阶段1：停车 200 ms。
                cmd = MoveCmd.STOP
            elif elapsed < 1200:
                # 阶段2：后退 1000 ms（累计时间 200~1200 ms）。
                cmd = MoveCmd.BACKWARD
            elif elapsed < 2000:
                # 阶段3：旋转 800 ms（累计时间 1200~2000 ms）。
                cmd = stuck_detector.escape_turn_cmd
            else:
                # 阶段4：退出 ESCAPE。持球重新寻找安全区，无球重新搜索目标。
                cmd = MoveCmd.STOP
                reset_transport_controller()
                grab_ready_count = 0
                grab_roi_ready_count = 0
                grab_close_start_time = 0
                grab_open_start_time = 0
                search_target_ready_count = 0
                stuck_detector.reset_observation()

                if CURRENT_HOLDING_TYPE != "NONE":
                    ROBOT_STATE = STATE_TRANSPORT
                    action_timer = 0
                    print(">>> ESCAPE complete: holding ball, back to TRANSPORT")
                else:
                    ROBOT_STATE = STATE_SEARCH
                    action_timer = 0
                    print(">>> ESCAPE complete: no ball, back to SEARCH")

            payload = make_normal_payload(cmd, grip)

        elif ROBOT_STATE == STATE_UNLOAD:
            elapsed = time.ticks_diff(time.ticks_ms(), action_timer)

            if elapsed < 500:
                cmd = MoveCmd.STOP
                grip = GripAction.GRAB
            elif elapsed < 2000:
                cmd = MoveCmd.FORWARD
                grip = GripAction.RELEASE
            elif elapsed < 3500:
                cmd = MoveCmd.BACKWARD
                grip = GripAction.RELEASE
            else:
                CURRENT_HOLDING_TYPE = "NONE"
                TRACK_LOCK_INFO = None
                LOCKED_TARGET_INFO = None
                unload_ready_count = 0
                search_target_ready_count = 0
                ROBOT_STATE = STATE_SEARCH
                reset_transport_controller()

                search_cooldown_timer = time.ticks_ms()

                cmd = MoveCmd.STOP
                grip = GripAction.RELEASE
                print(">>> unload complete: back to SEARCH with Cooldown")

            payload = make_normal_payload(cmd, grip)

        # ===================== 全场卡住监测与 ESCAPE 触发 =====================
        # 根据最终 payload 判断实际运动意图；TRACK/TRANSPORT 中外层 cmd 可能仍是 STOP。
        motion_now_ms = time.ticks_ms()
        actual_move_cmd = get_payload_move_cmd(payload)
        stuck_monitor_enabled = should_monitor_stuck(
            ROBOT_STATE,
            payload,
            safe_detected,
            safe_blob,
            grab_open_start_time,
        )

        if stuck_detector.update(cv_img, motion_now_ms, stuck_monitor_enabled):
            stuck_detector.choose_escape_turn(actual_move_cmd)
            print(">>> STUCK detected: state={} score={} move={} escape_turn={}".format(
                STATE_NAMES.get(ROBOT_STATE, "UNKNOWN"),
                stuck_detector.last_score,
                actual_move_cmd,
                stuck_detector.escape_turn_cmd,
            ))

            ROBOT_STATE = STATE_ESCAPE
            action_timer = motion_now_ms
            search_target_ready_count = 0
            grab_ready_count = 0
            grab_roi_ready_count = 0
            grab_close_start_time = 0
            grab_open_start_time = 0
            reset_transport_controller()
            stuck_detector.reset_observation()

            # 触发当帧立即停车，下一帧开始按 ESCAPE 时间轴执行。
            payload = make_normal_payload(MoveCmd.STOP, GripAction.GRAB)

        center_y_draw = SENSOR_HEIGHT // 2
        osd_img.draw_rectangle(GRIP_ROI, color=(0, 255, 0), thickness=2)
        draw_label(osd_img, GRIP_ROI[0], GRIP_ROI[1] - 25, "GRIP_ROI", (0, 255, 0))
        osd_img.draw_rectangle(CV_GRAB_ROI, color=(0, 255, 255), thickness=2)
        draw_label(
            osd_img, CV_GRAB_ROI[0], CV_GRAB_ROI[1] - 50,
            "CV_GRAB_ROI", (0, 255, 255)
        )
        if control_target is not None and control_target.get("source") == "CV2":
            cv_grab_x, cv_grab_y = get_cv_grab_point(control_target)
            osd_img.draw_cross(cv_grab_x, cv_grab_y, color=(0, 255, 255), size=12)
        if (
            ROBOT_STATE == STATE_VERIFY_GRAB and
            LOCKED_TARGET_INFO is not None and
            LOCKED_TARGET_INFO.get("source") == "CV2"
        ):
            osd_img.draw_rectangle(VERIFY_GRAB_ROI, color=(180, 0, 255), thickness=2)
            if cv_verify_info is not None and cv_verify_info.get("best_rect") is not None:
                osd_img.draw_rectangle(
                    cv_verify_info["best_rect"], color=(0, 255, 120), thickness=2
                )
        osd_img.draw_cross(CENTER_X, center_y_draw, color=(0, 255, 0), size=12)
        draw_label(osd_img, CENTER_X + 10, center_y_draw + 10, "CENTER_X", (0, 255, 0))

        # ==========================================================
        # OSD 屏幕左上角全局状态与真实 Sign 值显示
        current_state_name = STATE_NAMES.get(ROBOT_STATE, "UNKNOWN")
        actual_sign = payload[0]  # 取出当前帧实际要发送的 sign 值
        draw_label(osd_img, 10, 10, "STATE:{} SIGN:{}".format(current_state_name, actual_sign), (255, 255, 0))
        draw_label(osd_img, 10, 330, "MOTION:{} STILL:{}".format(
            stuck_detector.last_score,
            stuck_detector.still_elapsed(time.ticks_ms()),
        ), (0, 255, 255))
        draw_label(osd_img, 10, 362, "Y:{} CV:{} DB:{} DS:{} {}".format(
            len(yolo_targets), len(cv_targets), len(cv_dropped_ball),
            len(cv_dropped_safe), scheduler_mode
        ), (0, 255, 255))
        if (
            ROBOT_STATE == STATE_VERIFY_GRAB and
            LOCKED_TARGET_INFO is not None and
            LOCKED_TARGET_INFO.get("source") == "CV2"
        ):
            verify_raw = cv_verify_info["raw_pixels"] if cv_verify_info is not None else -1
            verify_type = LOCKED_TARGET_INFO["type"]
            draw_label(osd_img, 10, 394, "CVVERIFY:{} RAW:{} BASE:{} EX:{}".format(
                verify_type, verify_raw,
                empty_verify_baseline_pixels.get(verify_type), cv_verify_excess
            ), (0, 255, 255))

        if control_target:
            aim_x = control_target.get("aim_x", control_target["cx"])
            aim_y = control_target.get("aim_y", control_target["cy"])
            osd_img.draw_line(CENTER_X, center_y_draw, aim_x, aim_y, color=(0, 255, 255), thickness=2)
            osd_img.draw_cross(aim_x, aim_y, color=(255, 255, 255), size=12)
            err_x = aim_x - CENTER_X
            draw_label(osd_img, 10, 42, "EX:{} AX:{} AY:{}".format(err_x, aim_x, aim_y), (0, 255, 255))
            draw_label(osd_img, 10, 74, "SRC:{} TYPE:{}".format(
                control_target.get("source", "YOLO"), get_display_label(control_target["label"])
            ), (0, 255, 255))

        if ROBOT_STATE == STATE_TRACK:
            if grab_open_start_time == 0:
                track_grip_phase_name = "CLOSED_APPROACH"
                track_grip_detail = "OPEN_Y:{} CNT:{}/3".format(
                    TRACK_GRIP_OPEN_TARGET_Y, grab_ready_count
                )
            elif grab_open_start_time > 0:
                track_grip_phase_name = "OPENING"
                track_grip_detail = "OPEN_WAIT:500ms"
            else:
                track_grip_phase_name = "OPEN_APPROACH"
                if control_target is not None and control_target.get("source") == "CV2":
                    cv_grab_x, cv_grab_y = get_cv_grab_point(control_target)
                    track_grip_detail = "CV BTM:({}, {}) C:{}/3".format(
                        cv_grab_x, cv_grab_y, grab_ready_count
                    )
                else:
                    track_grip_detail = "YOLO GRAB_Y:{} C:{}/{}".format(
                        TRACK_GRAB_TARGET_Y, grab_ready_count, GRAB_CONFIRM_FRAMES
                    )
            draw_label(osd_img, 10, 106, "TRACK_GRIP:{}".format(track_grip_phase_name), (255, 0, 255))
            draw_label(osd_img, 10, 138, track_grip_detail, (255, 0, 255))

        if ROBOT_STATE == STATE_GRAB:
            if grab_close_start_time != 0:
                if LOCKED_TARGET_INFO is not None and LOCKED_TARGET_INFO.get("source") == "CV2":
                    grab_phase_name = "CV_CLOSING_DIRECT"
                else:
                    grab_phase_name = "CLOSING"
            else:
                grab_phase_name = "ROI_CONFIRM"
            draw_label(osd_img, 10, 106, "GRAB_PHASE:{}".format(grab_phase_name), (255, 0, 255))

        if ROBOT_STATE == STATE_VERIFY_GRAB:
            draw_label(osd_img, 10, 106, "VERIFY_GRAB:{}/{}".format(
                verify_grab_ready_count, VERIFY_GRAB_CONFIRM_FRAMES
            ), (255, 0, 255))
            verify_source = LOCKED_TARGET_INFO.get("source") if LOCKED_TARGET_INFO is not None else "NONE"
            draw_label(osd_img, 10, 138, "OK:{} SRC:{} COUNT:{} T:{}".format(
                verify_grab_debug_detected,
                verify_source,
                verify_grab_debug_target_count,
                verify_grab_debug_elapsed
            ), (255, 0, 255))

        if ROBOT_STATE == STATE_TRANSPORT:
            draw_label(osd_img, 10, 106, "TRANSPORT:SAFE_XY", (255, 0, 0))
            draw_label(osd_img, 10, 138, "safe_x:{}".format(transport_debug_safe_x), (255, 0, 0))
            draw_label(osd_img, 10, 170, "safe_y:{}".format(transport_debug_safe_y), (255, 0, 0))
            draw_label(osd_img, 10, 202, "safe_box_area:{}".format(transport_debug_box_area), (255, 0, 0))
            draw_label(osd_img, 10, 234, "xy_ready_count:{}".format(transport_debug_xy_ready_count), (255, 0, 0))

        if ROBOT_STATE == STATE_ESCAPE:
            escape_osd_elapsed = time.ticks_diff(time.ticks_ms(), action_timer)
            if escape_osd_elapsed < 200:
                escape_osd_phase = "STOP"
            elif escape_osd_elapsed < 1200:
                escape_osd_phase = "BACKWARD"
            elif escape_osd_elapsed < 2000:
                escape_osd_phase = "TURN"
            else:
                escape_osd_phase = "EXIT"
            draw_label(osd_img, 10, 106, "ESCAPE:{} {}ms".format(
                escape_osd_phase, escape_osd_elapsed
            ), (255, 0, 0))
            draw_label(osd_img, 10, 138, "TURN_CMD:{}".format(
                stuck_detector.escape_turn_cmd
            ), (255, 0, 0))
        # ==========================================================

        if debug_print_count % 15 == 0:
            print("FRAME={} STATE={} SCHED={} YRUN={} CRUN={} Y={} CV={} SAFE={}".format(
                frame_count, STATE_NAMES.get(ROBOT_STATE, "UNKNOWN"), scheduler_mode,
                run_yolo, run_cv, len(yolo_targets), len(cv_targets), safe_detected
            ))
            if (
                ROBOT_STATE == STATE_VERIFY_GRAB and
                LOCKED_TARGET_INFO is not None and
                LOCKED_TARGET_INFO.get("source") == "CV2"
            ):
                verify_type = LOCKED_TARGET_INFO["type"]
                verify_raw = cv_verify_info["raw_pixels"] if cv_verify_info is not None else -1
                verify_area = cv_verify_info["best_area"] if cv_verify_info is not None else 0
                print("  CVVERIFY type={} raw={} base={} excess={} area={:.0f}".format(
                    verify_type, verify_raw,
                    empty_verify_baseline_pixels.get(verify_type),
                    cv_verify_excess, verify_area
                ))
            if last_cv_reject_counts:
                print("  CV_REJECT:", last_cv_reject_counts)

        send_payload_if_needed(payload)
        Display.show_image(osd_img, 0, 0, Display.LAYER_OSD0)
        if current_hsv_img is not None:
            del current_hsv_img
        if yolo_input is not None:
            del yolo_input
        if yolo_img is not None:
            del yolo_img
        if cv_img is not None:
            del cv_img
        if debug_print_count % 30 == 0:
            gc.collect()
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
        Display.deinit()
    except Exception:
        pass
    try:
        MediaManager.deinit()
    except Exception:
        pass
    gc.collect()
