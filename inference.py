#!/usr/bin/env python3
# inference_cpu.py
"""
离线 CPU 推理脚本模板（支持 .pth 和 TorchScript .pt）
Usage examples:
  python inference_cpu.py --weights weights/LBA-Net_best.pth --input data/case.png --output outputs/
  python inference_cpu.py --weights weights/LBA-Net_best.pth --input data/images_dir/ --output outputs/ --batch
  python inference_cpu.py --weights weights/LBA-Net_cpu.pt --input data/video.mp4 --output outputs/ --video
"""

import os
import time
import argparse
from glob import glob
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# -------------------------------
# -------- USER SECTION ---------
# -------------------------------
# 1) 如果 model.py（包含类 LBANet 或其他），把它放在同一目录或在PYTHONPATH中。
#    然后把下面的 `from model import LBANet` 取消注释并把 ModelClass 替换成对应的类。
# 2) 使用 state_dict (.pth文件) 但没有模型定义，可以先用训练端导出 TorchScript: torch.jit.trace(...) 保存 .pt，
#    然后直接使用 --weights path/to/model.pt （TorchScript） 来推理（无需模型源代码）。
#本代码支持：
#1.从训练得到的 *.pth（state_dict）或 TorchScript (*.pt) 加载模型；
#2.单张图 / 文件夹批量 / 视频（或摄像头）推理；
#3.结果保存为分割 mask 和叠加可视化图；
#4.在 CPU 上测量并打印每张图像的推理时间；
#5.包含可选的 ONNX 导出与基本后处理（阈值 + 小结构过滤）示例；
#6.明确标注你需要替换/填入的模型定义位置。
#
# Example:
# from model import LBANet  # <-- 如果你有模型定义，取消注释并确保类名正确
ModelClass = None  # 如果使用自己的模型类，替换为 LBANet 或相应类；否则保持 None，脚本会尝试 torch.jit.load

# -------------------------------
# ----- End USER SECTION -------
# -------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Offline CPU inference for thyroid ultrasound segmentation")
    p.add_argument("--weights", required=True, help="Path to model weights (.pth state_dict) or TorchScript (.pt) or ONNX (.onnx)")
    p.add_argument("--input", required=True, help="Path to input image / folder / video / camera index (0,1,...)")
    p.add_argument("--output", default="outputs", help="Output folder to save masks and overlays")
    p.add_argument("--img-size", type=int, default=512, help="Square size to resize input to (default 512)")
    p.add_argument("--batch", action="store_true", help="Treat input as folder and run batch")
    p.add_argument("--video", action="store_true", help="Treat input as video (mp4, avi) or camera index")
    p.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    p.add_argument("--cpu-threads", type=int, default=4, help="Number of CPU threads to use")
    p.add_argument("--save-mask", action="store_true", help="Save binary mask png")
    p.add_argument("--save-overlay", action="store_true", help="Save overlay visualization png")
    p.add_argument("--to-onnx", action="store_true", help="Export model to ONNX (if weight is .pth and ModelClass provided)")
    return p.parse_args()


# -------------------------------
# Preprocess / Postprocess utils
# -------------------------------
def preprocess(img, img_size=512):
    """
    img: numpy array grayscale or BGR
    returns: numpy float32 array shape (1,1,H,W) normalized [0,1]
    """
    if img is None:
        return None
    # If BGR convert to gray
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    # resize keeping aspect ratio (here simple resize to square for simplicity)
    img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = img_resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)  # channel dim
    tensor = np.expand_dims(tensor, axis=0)  # batch dim
    return tensor  # shape (1,1,H,W)


def postprocess_mask(prob_map, threshold=0.5, min_area=100):
    """
    prob_map: numpy array HxW float [0,1]
    returns: binary mask (uint8 HxW), cleaned
    """
    mask = (prob_map >= threshold).astype(np.uint8) * 255
    # remove small components
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            cv2.drawContours(cleaned, [cnt], -1, 255, thickness=-1)
    return cleaned


def overlay_mask_on_image(img, mask, alpha=0.5):
    """
    img: original BGR image (H,W,3) or gray -> convert
    mask: binary mask uint8 HxW (0/255)
    returns: BGR blended image
    """
    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img.copy()
    color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, color_mask, alpha, 0)
    return blended


# -------------------------------
# Model loading utilities
# -------------------------------
def load_model(weights_path, device="cpu", model_class=None, img_size=512):
    """
    Attempts to load:
     - TorchScript .pt -> torch.jit.load
     - state_dict .pth -> requires model_class to be provided
     - ONNX (.onnx) -> will return special marker (user must use onnxruntime)
    Returns: model (torch.nn.Module ready in eval mode) or ("onnx", path)
    """
    ext = Path(weights_path).suffix.lower()
    if ext in [".pt", ".pth", ".bin"]:
        # if it's a TorchScript file (.pt) we can try to load it directly
        # Try TorchScript load first:
        try:
            # torch.jit.load works for both .pt scripted/traced models
            ts_model = torch.jit.load(weights_path, map_location=device)
            ts_model.eval()
            print(f"[INFO] Loaded TorchScript model from {weights_path}")
            return ts_model
        except Exception as e:
            print(f"[INFO] torch.jit.load failed: {e}. Trying state_dict load...")

        # state_dict path: need model class
        if model_class is None:
            raise RuntimeError("State_dict provided but no model class available. "
                               "Either provide ModelClass or export TorchScript (.pt) from training side.")
        # Instantiate model and load state_dict
        model = model_class()
        state = torch.load(weights_path, map_location=device)
        # if state is a dict with keys like 'model_state' adjust here as needed
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        print(f"[INFO] Loaded state_dict into model from {weights_path}")
        return model

    elif ext == ".onnx":
        print("[INFO] ONNX model detected. Use ONNX Runtime for inference.")
        return ("onnx", weights_path)
    else:
        raise RuntimeError(f"Unsupported model extension: {ext}")


# -------------------------------
# Main inference functions
# -------------------------------
def infer_on_image(model, img_np, img_size, device, threshold, save_mask=True, save_overlay=True, out_base=None):
    x = preprocess(img_np, img_size=img_size)
    if x is None:
        return None
    inp = torch.from_numpy(x).to(device)
    with torch.no_grad():
        start = time.time()
        # If model is TorchScript or nn.Module
        if isinstance(model, torch.jit.ScriptModule) or isinstance(model, torch.nn.Module):
            out = model(inp)
            # model may return tensor or (tensor, aux)
            if isinstance(out, (tuple, list)):
                seg = out[0]
            else:
                seg = out
            # assume output is logits (1x1xHxW) or similar
            seg = torch.sigmoid(seg)
            seg_np = seg.cpu().numpy()[0, 0]
        else:
            raise RuntimeError("Model type not supported in this runner (expected torch module/script).")
        elapsed = time.time() - start

    # resize seg_np back to original image size
    h_orig, w_orig = img_np.shape[:2]
    seg_resized = cv2.resize((seg_np * 255).astype(np.uint8), (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    prob_map = seg_resized.astype(np.float32) / 255.0
    mask = postprocess_mask(prob_map, threshold=threshold)
    overlay = overlay_mask_on_image(img_np, mask)

    # Save outputs
    if out_base:
        os.makedirs(os.path.dirname(out_base), exist_ok=True)
        if save_mask:
            mask_path = out_base + "_mask.png"
            cv2.imwrite(mask_path, mask)
        if save_overlay:
            over_path = out_base + "_overlay.png"
            cv2.imwrite(over_path, overlay)
    return {"time_s": elapsed, "mask": mask, "overlay": overlay, "prob_map": prob_map}


def run_batch(model, input_folder, out_folder, img_size, device, threshold, save_mask, save_overlay):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.dcm")
    files = []
    for e in exts:
        files += glob(os.path.join(input_folder, e))
    files = sorted(files)
    if len(files) == 0:
        print("[WARN] No image files found in folder:", input_folder)
        return

    times = []
    for f in files:
        print(f"[INFO] Processing {f} ...")
        # if DICOM support needed, add pydicom handling here
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        name = Path(f).stem
        out_base = os.path.join(out_folder, name)
        res = infer_on_image(model, img, img_size, device, threshold, save_mask, save_overlay, out_base)
        if res:
            print(f"  -> time {res['time_s']*1000:.1f} ms")
            times.append(res["time_s"])
    if times:
        print(f"[SUMMARY] Avg time per image: {np.mean(times)*1000:.1f} ms, median: {np.median(times)*1000:.1f} ms")


def run_video(model, source, out_folder, img_size, device, threshold, save_mask, save_overlay):
    # source can be path or camera index
    try:
        src = int(source)
    except:
        src = source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source:", source)
        return

    os.makedirs(out_folder, exist_ok=True)
    frame_id = 0
    times = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start_time = time.time()
        res = infer_on_image(model, gray, img_size, device, threshold, save_mask=False, save_overlay=False, out_base=None)
        times.append(res["time_s"])
        overlay = res["overlay"]
        fps = 1.0 / res["time_s"] if res["time_s"]>0 else 0.0

        # display
        disp = cv2.putText(overlay.copy(), f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.imshow("Inference", disp)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

        # optional save per-frame
        if save_overlay:
            cv2.imwrite(os.path.join(out_folder, f"frame_{frame_id:06d}_overlay.png"), overlay)
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    if times:
        print(f"[SUMMARY] Avg FPS: {1.0/np.mean(times):.2f}, Avg time per frame: {np.mean(times)*1000:.1f} ms")


# -------------------------------
# Optional: ONNX export helper
# -------------------------------
def export_to_onnx(model, weights_path, img_size=512, out_name="model.onnx"):
    dummy = torch.randn(1, 1, img_size, img_size)
    # if model is torchscript loaded we need the original nn.Module; skip if not available
    if isinstance(model, torch.jit.ScriptModule) or isinstance(model, torch.jit.ScriptFunction):
        print("[WARN] model appears to be TorchScript; ONNX export requires original nn.Module instance.")
        return
    torch.onnx.export(model, dummy, out_name, opset_version=13, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    print("[INFO] Exported to", out_name)


# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    # set CPU threads
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cpu")

    # attempt to import user model class if ModelClass is a string path? skipping dynamic import here.
    model = load_model(args.weights, device=device, model_class=ModelClass, img_size=args.img_size)

    # if ONNX model, fallback to user using ONNX Runtime (not implemented here)
    if isinstance(model, tuple) and model[0] == "onnx":
        print("[ERROR] Received ONNX model; this script uses PyTorch/ScriptModule inference. "
              "For ONNX runtime, use onnxruntime separately.")
        return

    # optionally export to onnx if requested and we have nn.Module
    if args.to_onnx:
        try:
            export_to_onnx(model, args.weights, img_size=args.img_size, out_name=os.path.join(args.output, "exported.onnx"))
        except Exception as e:
            print("[ERROR] ONNX export failed:", e)

    # Input handling
    if args.batch:
        # input must be folder
        if not os.path.isdir(args.input):
            raise RuntimeError("Batch mode requires input folder path")
        run_batch(model, args.input, args.output, args.img_size, device, args.threshold, args.save_mask, args.save_overlay)

    elif args.video:
        run_video(model, args.input, args.output, args.img_size, device, args.threshold, args.save_mask, args.save_overlay)

    else:
        # single image
        if not os.path.exists(args.input):
            raise RuntimeError("Input path does not exist")
        img = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("Failed to read image (are you sure it's an image?)")
        name = Path(args.input).stem
        out_base = os.path.join(args.output, name)
        res = infer_on_image(model, img, args.img_size, device, args.threshold, args.save_mask, args.save_overlay, out_base)
        print(f"[INFO] Done. Time: {res['time_s']*1000:.1f} ms, results saved to {args.output}")


if __name__ == "__main__":
    main()

