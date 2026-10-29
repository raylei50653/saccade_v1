"""Train YOLOv26 baseline / BiVisionMamba / DualMamba. Runtime patch, no source mod."""
import os
import sys
import inspect
import textwrap
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def register_custom_modules():
    """Runtime-patch ultralytics' parse_model to recognize BiVisionMamba,
    DualMambaBlock, and MambaPSA."""
    from bivision_mamba import BiVisionMamba
    from dual_mamba import DualMambaBlock
    from mamba_psa import MambaPSA
    import ultralytics.nn.tasks as tasks

    if getattr(tasks, "_custom_patched", False):
        return

    tasks.BiVisionMamba = BiVisionMamba
    tasks.DualMambaBlock = DualMambaBlock
    tasks.MambaPSA = MambaPSA

    src = textwrap.dedent(inspect.getsource(tasks.parse_model))

    if "MambaPSA" not in src:
        for old in [
            "base_modules = frozenset({",
            "base_modules = frozenset(\n        {",
            "base_modules = frozenset(\n    {",
        ]:
            if old in src:
                src = src.replace(
                    old,
                    old + "BiVisionMamba, DualMambaBlock, MambaPSA, ",
                    1,
                )
                break
        else:
            raise RuntimeError("base_modules not found")

    local_ns = {}
    exec(src, tasks.__dict__, local_ns)
    tasks.parse_model = local_ns["parse_model"]
    tasks._custom_patched = True
    print("[OK] BiVisionMamba + DualMambaBlock + MambaPSA registered.")


register_custom_modules()

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26n.yaml")
    p.add_argument("--data", default="VOC.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--name", default="exp")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=0,             # WSL2 keep at 0
        optimizer="AdamW",
        lr0=0.001,
        warmup_epochs=5,
        close_mosaic=15,
        amp=True,
        project="runs",
        name=args.name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()