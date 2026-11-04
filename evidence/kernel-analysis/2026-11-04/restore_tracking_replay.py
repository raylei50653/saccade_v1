#!/usr/bin/env python3
"""Restore only frame 20 of the pinned native-ABI replay into a new directory.

Earlier slots are zero placeholders, not valid recorded frames. tracking_probe
reads only frame 20. The original complete replay is not reconstructed.
"""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

bundle = json.loads(Path(__file__).with_name("tracking-selected-records.json").read_text())
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
for record in bundle["records"]:
    data = zlib.decompress(base64.b64decode(record["zlib_base64"]))
    assert len(data) == record["record_bytes"]
    assert hashlib.sha256(data).hexdigest() == record["sha256"]
    with (out / record["filename"]).open("wb") as stream:
        stream.seek(bundle["frame"] * len(data))
        stream.write(data)
print("Restored frame", bundle["frame"], "only to", out)
