# 自动采图 + 官方 YOLO 实时画框 + 同时保存原图/有框图
# K230 CanMV
#
# 保存结果：
#   /data/dataset/raw/    原始无框 RGBP888 .bin
#   /data/dataset/boxed/  已写入检测框的 RGBP888 .bin
#
# 两个目录中的同名文件是一一对应的，原有 BIN -> JPG 转换脚本可分别处理两个目录。

import os
import time
import gc
import image

from media.sensor import *
from media.display import *
from media.media import *
from libs.PipeLine import PipeLine
from libs.YOLO import YOLOv5

# =========================
# 参数区
# =========================
CAMERA_ID = 1
IMG_WIDTH = 800
IMG_HEIGHT = 480

SAVE_DIR = "/data/dataset"
RAW_SAVE_DIR = SAVE_DIR + "/raw"
BOXED_SAVE_DIR = SAVE_DIR + "/boxed"

IMG_PREFIX = "img_6.22_"
IMG_EXT = ".bin"
NUM_WIDTH = 5

CAPTURE_PER_SECOND = 2
CAPTURE_INTERVAL_MS = int(1000 / CAPTURE_PER_SECOND)

KMODEL_PATH = "/sdcard/model/best_0719.kmodel"
MODEL_INPUT_SIZE = [320, 320]
LABELS = [
    "redball", "blueball", "redboard", "blueboard", "redsection",
    "bluesection", "shizi", "redcube", "bluecube", "greenzhu",
    "yellowtai", "blackball", "yellowball",
]
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.45
MAX_BOXES_NUM = 50


def ensure_dir(path):
    """逐级创建目录，兼容 CanMV MicroPython。"""
    parts = path.split("/")
    cur = ""
    for part in parts:
        if part == "":
            continue
        cur += "/" + part
        try:
            os.stat(cur)
        except Exception:
            os.mkdir(cur)


def parse_index_from_name(name):
    """从 img_6.22_00001.bin 中提取编号。"""
    if not name.startswith(IMG_PREFIX) or not name.endswith(IMG_EXT):
        return None

    try:
        number_text = name[len(IMG_PREFIX):-len(IMG_EXT)]
        return int(number_text)
    except Exception:
        return None


def get_next_index(search_dirs):
    """
    同时扫描旧目录、raw 目录和 boxed 目录，避免重新从 00001 开始。
    """
    max_id = 0

    for directory in search_dirs:
        try:
            files = os.listdir(directory)
        except Exception:
            continue

        for name in files:
            file_id = parse_index_from_name(name)
            if file_id is not None and file_id > max_id:
                max_id = file_id

    return max_id + 1


def make_file_name(index):
    # 保持与原代码一致的五位编号格式。
    return "{}{:05d}{}".format(IMG_PREFIX, index, IMG_EXT)


def save_binary(data, file_path):
    """把 ndarray/缓冲区按原始字节写入 BIN。"""
    with open(file_path, "wb") as file_obj:
        try:
            file_obj.write(data)
        except Exception:
            file_obj.write(data.tobytes())


def get_display_label(label):
    name_map = {
        "redball": "RED",
        "blueball": "BLUE",
        "blackball": "CORE",
        "yellowball": "DANGER",
        "redboard": "REDBOARD",
        "blueboard": "BLUEBOARD",
    }
    return name_map.get(label, str(label))


def parse_yolo_results(res):
    """
    将 YOLOv5 检测结果转换成统一字典格式。

    主要兼容 CanMV 官方检测结果：
        [[x1, y1, x2, y2, score, class_id], ...]

    同时保留对以下格式的兼容：
        (boxes, class_ids, scores)
        [{"x": ..., "y": ..., "w": ..., "h": ...}, ...]
    """
    dets = []
    if res is None:
        return dets

    def append_det(x1, y1, x2, y2, score, class_id=None, label=None):
        if label is None and class_id is not None:
            try:
                class_id = int(class_id)
                if 0 <= class_id < len(LABELS):
                    label = LABELS[class_id]
            except Exception:
                label = None

        try:
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # 少数自定义结果可能返回 x、y、w、h。
            if x2 <= x1 or y2 <= y1:
                x2 = x1 + max(0, x2)
                y2 = y1 + max(0, y2)

            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w == 0 or h == 0:
                return

            dets.append({
                "x1": x1,
                "y1": y1,
                "rect": (x1, y1, w, h),
                "cx": x1 + w // 2,
                "cy": y1 + h // 2,
                "score": float(score) if score is not None else 0.0,
                "label": label,
            })
        except Exception:
            pass

    # 格式一：(boxes, class_ids, scores)
    # 必须确认 boxes 是二维框列表，避免把“恰好有 3 个检测框”的结果误判成三元组。
    if isinstance(res, tuple) and len(res) == 3:
        boxes, class_ids, scores = res[0], res[1], res[2]
        try:
            is_box_matrix = (
                isinstance(boxes, (tuple, list))
                and len(boxes) > 0
                and isinstance(boxes[0], (tuple, list))
                and len(boxes[0]) >= 4
            )
        except Exception:
            is_box_matrix = False

        if is_box_matrix and isinstance(class_ids, (tuple, list)):
            for i in range(min(len(boxes), len(class_ids))):
                score = scores[i] if isinstance(scores, (tuple, list)) else 0.0
                append_det(
                    boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                    score, class_ids[i], None
                )
            return dets

    # 格式二：CanMV 官方检测行列表，通常为：
    # [x1, y1, x2, y2, score, class_id]
    try:
        for entry in res:
            if entry is None:
                continue

            if isinstance(entry, dict):
                x = entry.get("x1", entry.get("x", 0))
                y = entry.get("y1", entry.get("y", 0))

                if "x2" in entry and "y2" in entry:
                    x2 = entry.get("x2")
                    y2 = entry.get("y2")
                else:
                    x2 = x + entry.get("w", 0)
                    y2 = y + entry.get("h", 0)

                score = entry.get("score", entry.get("confidence", 0.0))
                class_id = entry.get("class_id", entry.get("classid"))
                label = entry.get("label")
                append_det(x, y, x2, y2, score, class_id, label)
                continue

            if isinstance(entry, (tuple, list)) and len(entry) >= 6:
                append_det(
                    entry[0], entry[1], entry[2], entry[3],
                    entry[4], entry[5], None
                )
    except Exception:
        pass

    return dets

def get_box_color(label):
    label_str = str(label)
    if "red" in label_str:
        return (255, 0, 0)
    if "blue" in label_str:
        return (0, 0, 255)
    if "yellow" in label_str:
        return (255, 255, 0)
    if "black" in label_str:
        return (255, 255, 255)
    return (0, 255, 0)


def draw_yolo_detections(target_img, detections):
    """
    可同时用于：
      1. pl.osd_img：屏幕 OSD 层
      2. RGBP888 Image：真正写进待保存照片的像素
    """
    for det in detections:
        color = get_box_color(det["label"])

        target_img.draw_rectangle(det["rect"], color=color, thickness=2)
        target_img.draw_cross(
            det["cx"], det["cy"], color=color, size=8, thickness=2
        )

        text = "{} {:.2f}".format(
            get_display_label(det["label"]), det["score"]
        )
        text_y = max(0, det["y1"] - 24)

        # RGBP888 上部分固件可能不支持高级字体，失败时退回普通 ASCII 绘字。
        try:
            target_img.draw_string_advanced(
                det["x1"], text_y, 24, text, color
            )
        except Exception:
            try:
                target_img.draw_string(
                    det["x1"], text_y, text, color=color, scale=2
                )
            except Exception:
                pass


def draw_detections_on_rgbp_array(img_np, detections):
    """
    RGBP888 Image 绘图接口不可用时的后备方案。

    直接在 RGBP888 ndarray 上画矩形和中心十字，不画文字。
    常见 shape 为 (1, 3, H, W)，同时兼容 (3, H, W) 和 (H, W, 3)。
    """
    shape = img_np.shape

    if len(shape) == 4:
        height = shape[2]
        width = shape[3]

        def set_pixel(x, y, color):
            img_np[0, 0, y, x] = color[0]
            img_np[0, 1, y, x] = color[1]
            img_np[0, 2, y, x] = color[2]

    elif len(shape) == 3 and shape[0] == 3:
        height = shape[1]
        width = shape[2]

        def set_pixel(x, y, color):
            img_np[0, y, x] = color[0]
            img_np[1, y, x] = color[1]
            img_np[2, y, x] = color[2]

    elif len(shape) == 3 and shape[2] == 3:
        height = shape[0]
        width = shape[1]

        def set_pixel(x, y, color):
            img_np[y, x, 0] = color[0]
            img_np[y, x, 1] = color[1]
            img_np[y, x, 2] = color[2]

    else:
        raise ValueError("Unsupported image ndarray shape: {}".format(shape))

    def clamp(value, lower, upper):
        if value < lower:
            return lower
        if value > upper:
            return upper
        return value

    for det in detections:
        x, y, w, h = det["rect"]
        x1 = clamp(int(x), 0, width - 1)
        y1 = clamp(int(y), 0, height - 1)
        x2 = clamp(int(x + w - 1), 0, width - 1)
        y2 = clamp(int(y + h - 1), 0, height - 1)
        color = get_box_color(det["label"])

        # 两像素粗矩形。
        for thickness_offset in range(2):
            top = clamp(y1 + thickness_offset, 0, height - 1)
            bottom = clamp(y2 - thickness_offset, 0, height - 1)
            left = clamp(x1 + thickness_offset, 0, width - 1)
            right = clamp(x2 - thickness_offset, 0, width - 1)

            for px in range(left, right + 1):
                set_pixel(px, top, color)
                set_pixel(px, bottom, color)

            for py in range(top, bottom + 1):
                set_pixel(left, py, color)
                set_pixel(right, py, color)

        # 中心十字。
        cx = clamp(int(det["cx"]), 0, width - 1)
        cy = clamp(int(det["cy"]), 0, height - 1)
        for delta in range(-8, 9):
            px = cx + delta
            py = cy + delta
            if 0 <= px < width:
                set_pixel(px, cy, color)
            if 0 <= py < height:
                set_pixel(cx, py, color)


def save_frame_pair(img_np, yolo, result, detections, index):
    """
    保存同一帧的无框版和有框版。

    屏幕显示使用独立的 OSD 层；这里保存有框版时，才把框真正写入
    当前 RGBP888 帧的像素中。此函数应放在 pl.show_image() 之后调用，
    避免保存操作影响这一帧的实时预览。
    """
    file_name = make_file_name(index)
    raw_path = RAW_SAVE_DIR + "/" + file_name
    boxed_path = BOXED_SAVE_DIR + "/" + file_name

    boxed_img = None
    try:
        # 1. 原始无框 BIN。
        save_binary(img_np, raw_path)

        # 2. 将当前 RGBP888 ndarray 包装成 Image，共享同一块数据。
        boxed_img = image.Image(
            IMG_WIDTH,
            IMG_HEIGHT,
            image.RGBP888,
            alloc=image.ALLOC_REF,
            data=img_np,
        )

        draw_ok = False

        # 优先调用 CanMV YOLO 官方绘制方法。
        # 它直接认识 yolo.run() 的原始返回格式，不依赖我们手动解析。
        if result is not None:
            try:
                yolo.draw_result(result, boxed_img)
                draw_ok = True
            except Exception as native_draw_error:
                print("BOXED native draw fallback:", native_draw_error)

        # 部分固件可能不支持在 RGBP888 Image 上使用官方 draw_result，
        # 此时退回自定义绘制。
        if not draw_ok:
            try:
                draw_yolo_detections(boxed_img, detections)
                draw_ok = True
            except Exception as image_draw_error:
                print("RGBP888 image draw fallback:", image_draw_error)

        # 最后后备：直接修改 RGBP888 ndarray 的通道数据。
        if not draw_ok:
            draw_detections_on_rgbp_array(img_np, detections)

        # 3. 有框 BIN，通道排列仍为 RGBP888。
        save_binary(img_np, boxed_path)

    except Exception:
        # 任意一份保存失败时清理残留文件，避免 raw/boxed 编号不配对。
        try:
            os.remove(raw_path)
        except Exception:
            pass
        try:
            os.remove(boxed_path)
        except Exception:
            pass
        raise
    finally:
        if boxed_img is not None:
            del boxed_img

    return raw_path, boxed_path

def main():
    print(">>> 启动双版本采图模式：实时框 + RAW + BOXED")
    os.exitpoint(os.EXITPOINT_ENABLE)

    ensure_dir(SAVE_DIR)
    ensure_dir(RAW_SAVE_DIR)
    ensure_dir(BOXED_SAVE_DIR)

    # SAVE_DIR 用于兼容你之前直接保存在 /data/dataset 下的旧 BIN。
    next_id = get_next_index([SAVE_DIR, RAW_SAVE_DIR, BOXED_SAVE_DIR])
    print(">>> 下一个文件编号 =", next_id)
    print(">>> RAW目录   =", RAW_SAVE_DIR)
    print(">>> BOXED目录 =", BOXED_SAVE_DIR)

    pl = None
    sensor = None
    yolo = None

    # 继续沿用原来稳定的 PipeLine 架构。
    pl = PipeLine(
        rgb888p_size=[IMG_WIDTH, IMG_HEIGHT],
        display_mode="lcd",
        display_size=[IMG_WIDTH, IMG_HEIGHT],
    )
    sensor = Sensor(id=CAMERA_ID)
    pl.create(sensor)
    display_size = pl.get_display_size()

    yolo = YOLOv5(
        task_type="detect",
        mode="video",
        kmodel_path=KMODEL_PATH,
        labels=LABELS,
        rgb888p_size=[IMG_WIDTH, IMG_HEIGHT],
        model_input_size=MODEL_INPUT_SIZE,
        display_size=display_size,
        conf_thresh=CONFIDENCE_THRESHOLD,
        nms_thresh=NMS_THRESHOLD,
        max_boxes_num=MAX_BOXES_NUM,
        debug_mode=0,
    )
    yolo.config_preprocess()

    last_capture_ms = time.ticks_ms()
    saved_pair_count = 0

    try:
        while True:
            os.exitpoint()

            # RGBP888 ulab.ndarray，供模型推理和 BIN 保存使用。
            img = pl.get_frame()

            osd_img = pl.osd_img
            try:
                osd_img.clear()
            except Exception:
                pass

            # 本帧 YOLO 推理。
            result = None
            detections = []
            try:
                result = yolo.run(img)
                detections = parse_yolo_results(result)

                # 【实时显示框的关键】
                # 直接使用 CanMV YOLO 官方绘制函数，把本帧结果画到 OSD 层。
                # 官方例程也是 yolo.run -> yolo.draw_result -> pl.show_image。
                yolo.draw_result(result, osd_img)

            except Exception as infer_error:
                print("YOLO/Live Draw Error:", infer_error)

                # 若某个自定义 YOLO 库没有 draw_result，则用手动画框兜底。
                try:
                    draw_yolo_detections(osd_img, detections)
                except Exception:
                    pass

            # 状态文字也在 OSD 层，因此不会写进 raw 原图。
            try:
                osd_img.draw_string_advanced(
                    10,
                    10,
                    30,
                    "[Recording] Saved: {}  Det: {}".format(
                        saved_pair_count, len(detections)
                    ),
                    (0, 255, 0),
                )
            except Exception:
                pass

            # 先显示实时画面和检测框。
            pl.show_image()

            # 再保存当前帧。这样即使有框 BIN 需要修改 img 的像素，
            # 也不会破坏刚刚送去显示的 OSD 实时预览。
            now_ms = time.ticks_ms()
            if time.ticks_diff(now_ms, last_capture_ms) >= CAPTURE_INTERVAL_MS:
                last_capture_ms = now_ms

                try:
                    raw_path, boxed_path = save_frame_pair(
                        img, yolo, result, detections, next_id
                    )
                    saved_pair_count += 1
                    next_id += 1

                    print("Saved RAW   :", raw_path)
                    print("Saved BOXED :", boxed_path)
                except Exception as save_error:
                    print("Save Pair Error:", save_error)

            gc.collect()

    except KeyboardInterrupt:
        print("user stop")
    except BaseException as run_error:
        print("run error:", run_error)
    finally:
        print("stopping...")
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
            if pl is not None:
                pl.destroy()
        except Exception:
            pass
        try:
            MediaManager.deinit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
