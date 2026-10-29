"""Validate a YOLOv26 model with custom modules registered."""
import sys
import os
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 這行會 import train.py 並執行 register_custom_modules()
import train  # noqa: F401

from ultralytics import YOLO


p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data", default="VOC.yaml")
p.add_argument("--batch", type=int, default=16)
args = p.parse_args()

YOLO(args.model).val(
    data=args.data,
    workers=0,
    verbose=True,
    device=0,
    batch=args.batch,
)