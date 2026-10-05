"""Export the PilotNet steering model to TensorFlow SavedModel and TFLite (FP32 and INT8), and check them.

    python -m ads.export.tflite --model runs/steering/pilotnet/best.pt --out runs/export/pilotnet

The network is rebuilt in Keras layer for layer and the PyTorch weights are copied in: convolution
kernels go from (out, in, h, w) to (h, w, in, out), and the first dense layer's columns are reordered
because PyTorch flattens a (C, H, W) feature map and Keras an (H, W, C) one. Parity with PyTorch is
checked on random inputs before export.

INT8 is full-integer quantization of weights and activations, calibrated on training frames (not
validation frames), with float32 inputs and outputs. Every format is scored on the validation set, so
the accuracy cost of each step is measured, not assumed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

FEATURE_C, FEATURE_H, FEATURE_W = 64, 1, 18  # PilotNet's conv output for a 66x200 input


def build_keras(dropout: float = 0.2):
    import keras
    from keras import layers

    from ads import config

    image = keras.Input(shape=(config.NET_H, config.NET_W, 3), name="image")  # YUV, values 0..255, NHWC
    speed = keras.Input(shape=(1,), name="speed_kmh")
    x = layers.Rescaling(1 / 127.5, offset=-1.0)(image)
    for filters, k, s in ((24, 5, 2), (36, 5, 2), (48, 5, 2), (64, 3, 1), (64, 3, 1)):
        x = layers.Conv2D(filters, k, strides=s, activation="elu")(x)
    x = layers.Flatten()(x)
    s = layers.Rescaling(1 / config.SPEED_SCALE_KMH)(speed)
    x = layers.Concatenate()([x, s])
    x = layers.Dense(100, activation="elu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(50, activation="elu")(x)
    x = layers.Dense(10, activation="elu")(x)
    out = layers.Dense(1, name="steering")(x)
    return keras.Model([image, speed], out, name="pilotnet")


def copy_weights(torch_model, keras_model) -> None:
    convs = [m for m in torch_model.features if m.__class__.__name__ == "Conv2d"]
    linears = [m for m in torch_model.head if m.__class__.__name__ == "Linear"]
    k_convs = [layer for layer in keras_model.layers if layer.__class__.__name__ == "Conv2D"]
    k_dense = [layer for layer in keras_model.layers if layer.__class__.__name__ == "Dense"]
    for t, k in zip(convs, k_convs, strict=True):
        k.set_weights([t.weight.detach().numpy().transpose(2, 3, 1, 0), t.bias.detach().numpy()])
    # PyTorch flattens (C, H, W): column c*H*W + h*W + w. Keras flattens (H, W, C): row (h*W + w)*C + c.
    order = [
        c * FEATURE_H * FEATURE_W + h * FEATURE_W + w
        for h in range(FEATURE_H)
        for w in range(FEATURE_W)
        for c in range(FEATURE_C)
    ]
    first = linears[0].weight.detach().numpy()  # (100, 1152 + 1); the speed input is the last column in both
    w0 = np.concatenate([first[:, order], first[:, -1:]], axis=1).T
    k_dense[0].set_weights([w0, linears[0].bias.detach().numpy()])
    for t, k in zip(linears[1:], k_dense[1:], strict=True):
        k.set_weights([t.weight.detach().numpy().T, t.bias.detach().numpy()])


def torch_predict(model, images_nhwc: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(images_nhwc.transpose(0, 3, 1, 2)))
        return model(x, torch.from_numpy(speeds.astype(np.float32))).numpy()


def tflite_predict(path: Path, images_nhwc: np.ndarray, speeds: np.ndarray, threads: int = 1) -> np.ndarray:
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
    interp.allocate_tensors()
    inputs = {d["name"]: d["index"] for d in interp.get_input_details()}
    img_idx = next(v for k, v in inputs.items() if "image" in k)
    spd_idx = next(v for k, v in inputs.items() if "speed" in k)
    out_idx = interp.get_output_details()[0]["index"]
    preds = np.empty(len(images_nhwc), dtype=np.float32)
    for i in range(len(images_nhwc)):
        interp.set_tensor(img_idx, images_nhwc[i : i + 1].astype(np.float32))
        interp.set_tensor(spd_idx, speeds[i : i + 1].reshape(1, 1).astype(np.float32))
        interp.invoke()
        preds[i] = interp.get_tensor(out_idx).reshape(-1)[0]
    return preds


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Export PilotNet to SavedModel and TFLite FP32/INT8.")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--train-cache", type=Path, default=Path("data/cache/train"))
    p.add_argument("--val-cache", type=Path, default=Path("data/cache/val"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--calibration", type=int, default=500, help="training frames for INT8 calibration")
    p.add_argument("--val-limit", type=int, default=None)
    args = p.parse_args(argv)

    import tensorflow as tf

    from ads import config, provenance
    from ads.eval.closed_loop import load_model
    from ads.eval.metrics import split_mae

    args.out.mkdir(parents=True, exist_ok=True)
    torch_model = load_model(args.model)
    keras_model = build_keras()
    copy_weights(torch_model, keras_model)

    rng = np.random.default_rng(0)
    probe = rng.integers(0, 256, (16, config.NET_H, config.NET_W, 3)).astype(np.uint8)
    probe_speed = rng.uniform(0, 60, 16).astype(np.float32)
    k_out = keras_model.predict([probe.astype(np.float32), probe_speed.reshape(-1, 1)], verbose=0).reshape(-1)
    parity = float(np.abs(k_out - torch_predict(torch_model, probe, probe_speed)).max())
    if parity > 1e-4:
        raise SystemExit(f"Keras and PyTorch disagree by {parity:.2e}; weight copy is wrong")

    saved = args.out / "saved_model"
    keras_model.export(str(saved), format="tf_saved_model")

    def convert(int8: bool) -> bytes:
        conv = tf.lite.TFLiteConverter.from_saved_model(str(saved))
        if int8:
            train_imgs = np.load(args.train_cache / "images.npy", mmap_mode="r")
            train_speed = np.load(args.train_cache / "meta.npz")["speed_kmh"]
            pick = np.sort(rng.choice(len(train_imgs), size=args.calibration, replace=False))

            def representative():
                for i in pick:
                    yield {"image": train_imgs[i : i + 1].astype(np.float32), "speed_kmh": train_speed[i : i + 1].reshape(1, 1)}

            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = representative
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        return conv.convert()

    files = {"fp32": args.out / "pilotnet_fp32.tflite", "int8": args.out / "pilotnet_int8.tflite"}
    for name, path in files.items():
        path.write_bytes(convert(int8=name == "int8"))

    val_imgs = np.load(args.val_cache / "images.npy", mmap_mode="r")
    meta = np.load(args.val_cache / "meta.npz")
    n = min(args.val_limit or len(val_imgs), len(val_imgs))
    imgs, speeds = np.asarray(val_imgs[:n]), meta["speed_kmh"][:n]
    labels, curv = meta["steering"][:n], meta["curvature"][:n]
    ref = np.clip(torch_predict(torch_model, imgs, speeds), -1, 1)
    report = {
        "provenance": provenance.stamp("ads.export.tflite"),
        "model": str(args.model),
        "keras_vs_torch_max_abs": parity,
        "val_frames": int(n),
        "calibration_frames": args.calibration,
        "pytorch": split_mae(ref, labels, curv, config.CURVE_CURVATURE),
        "saved_model_bytes": sum(f.stat().st_size for f in saved.rglob("*") if f.is_file()),
    }
    for name, path in files.items():
        pred = np.clip(tflite_predict(path, imgs, speeds), -1, 1)
        report[name] = {
            "bytes": path.stat().st_size,
            **split_mae(pred, labels, curv, config.CURVE_CURVATURE),
            "max_abs_vs_pytorch": float(np.abs(pred - ref).max()),
            "mean_abs_vs_pytorch": float(np.abs(pred - ref).mean()),
        }
    (args.out / "export_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=1))
    return report


if __name__ == "__main__":
    main()
