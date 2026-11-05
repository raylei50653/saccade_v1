#!/usr/bin/env python3
"""Bounded native CLI integration; reads nine images, never MOT ground truth.

Usage: track_stream_cli_tests.py TRACK_STREAM HR_ENGINE MOT17_10_DIR
       [--output-root BUILD_DIR] [--ffmpeg PATH]

Exercises explicit CPU decode and the default NVDEC path. Artifacts are retained
in a new directory for every invocation. Native CUDA tracking uses fourteen short
sessions; H.264/HEVC fixtures include B frames and SDR BT.709 metadata.
"""

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time


PACKET_BYTES = 12368


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def packets(path, expected_count=None):
    """Independent v1 wire checks, including reserved/unused bytes and floats."""
    data = path.read_bytes()
    check(data and len(data) % PACKET_BYTES == 0, f"truncated/empty packets: {path.name}")
    count = len(data) // PACKET_BYTES
    if expected_count is not None:
        check(count == expected_count, f"packet count {count} != {expected_count}")
    observations = []
    for index in range(count):
        packet = data[index * PACKET_BYTES:(index + 1) * PACKET_BYTES]
        header = struct.unpack_from("<4sHHIIQqqqIIII", packet)
        check(header[:5] == (b"SCRP", 1, 80, PACKET_BYTES, 256), "packet profile")
        epoch, frame, committed, failure, status, rows, error, sticky = header[5:]
        check(epoch == 1 and frame == index, "packet identity/order")
        check(committed == frame and failure == -1 and status == error == sticky == 0, "non-OK packet")
        check(rows <= 256, "row capacity")
        check(not any(packet[64:80]), "nonzero header reserved bytes")
        check(not any(packet[80 + rows * 48:]), "nonzero unused rows")
        for row_index in range(rows):
            offset = 80 + row_index * 48
            uid, output_id, _, flags, *values = struct.unpack_from("<QQII6f", packet, offset)
            check(uid > 0 and output_id > 0 and flags in (1, 2), "row identity/flags")
            check(all(math.isfinite(value) for value in values), "nonfinite packet value")
            check(0 <= values[4] <= 1, "observation score")
            if flags == 2:
                check(values[4] == values[5] == 0, "predicted observation score")
            for component in range(6):
                check(struct.unpack_from("<I", packet, offset + 24 + 4 * component)[0] != 0x80000000,
                      "noncanonical negative zero")
            observations.append((frame + 1, output_id))
    return data, sorted(observations)


def check_mot(path, observations):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split(",")
        check(len(fields) == 10 and fields[7:] == ["-1", "-1", "-1"], "MOT columns")
        check(all(math.isfinite(float(value)) for value in fields[2:7]), "MOT values")
        rows.append((int(fields[0]), int(fields[1])))
    check(rows == observations, "online MOT must match packet frame/native output IDs")


class Harness:
    def __init__(self, executable, engine, output):
        self.executable, self.engine, self.output = executable, engine, output
        self.results = []

    def command(self, source, mot, *arguments, decode="cpu"):
        command = [str(self.executable), str(self.engine), str(source), str(mot)]
        if decode is not None:
            command += ["--decode", decode]
        return [*command, *map(str, arguments)]

    def run(self, name, command, expected=0, timeout=180):
        start = time.monotonic()
        completed = subprocess.run(command, capture_output=True, timeout=timeout)
        (self.output / f"{name}.stdout").write_bytes(completed.stdout)
        (self.output / f"{name}.stderr").write_bytes(completed.stderr)
        self.results.append({"case": name, "returncode": completed.returncode,
                             "elapsed_s": time.monotonic() - start})
        check(completed.returncode == expected,
              f"{name}: exit {completed.returncode} != {expected}; see retained stdout/stderr")
        return completed

    def interrupted(self, signum):
        name = signal.Signals(signum).name.lower()
        mot = self.output / f"{name}.txt"
        raw = self.output / f"{name}.bin"
        command = self.command("testsrc2=size=320x240:rate=10", mot,
                               "--format", "lavfi", "--packets", raw)
        start = time.monotonic()
        with (self.output / f"{name}.stdout").open("wb") as stdout, \
                (self.output / f"{name}.stderr").open("wb") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            try:
                deadline = time.monotonic() + 120
                while not (raw.exists() and raw.stat().st_size >= PACKET_BYTES):
                    check(process.poll() is None, f"{name}: exited before first output")
                    check(time.monotonic() < deadline, f"{name}: first-output timeout")
                    time.sleep(0.02)
                process.send_signal(signum)
                returncode = process.wait(timeout=30)
                check(returncode == 128 + signum, f"{name}: cancellation exit {returncode}")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
        _, observations = packets(raw)
        check_mot(mot, observations)
        summary = (self.output / f"{name}.stdout").read_text()
        check("end=cancelled" in summary, f"{name}: missing cancelled status")
        self.results.append({"case": name, "returncode": returncode,
                             "packets": raw.stat().st_size // PACKET_BYTES,
                             "elapsed_s": time.monotonic() - start})

    def open_interrupted(self):
        """Interrupt a constructor blocked before it can expose request_stop."""
        name = "sigint-during-open"
        mot, raw = self.output / f"{name}.txt", self.output / f"{name}.bin"
        accepted, release = threading.Event(), threading.Event()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(5)
        port = listener.getsockname()[1]

        def hold_response():
            try:
                connection, _ = listener.accept()
                with connection:
                    accepted.set()
                    release.wait(10)
            except OSError:
                pass
            finally:
                listener.close()

        server = threading.Thread(target=hold_response)
        server.start()
        secret = "test-only-open-cancellation-secret"
        uri = f"http://tester:{secret}@127.0.0.1:{port}/stalled"
        process = None
        start = time.monotonic()
        try:
            with (self.output / f"{name}.stdout").open("wb") as stdout, \
                    (self.output / f"{name}.stderr").open("wb") as stderr:
                process = subprocess.Popen(self.command(uri, mot, "--packets", raw, "--timeout-ms", "500"),
                                           stdout=stdout, stderr=stderr)
                check(accepted.wait(5), "source did not enter constructor HTTP open")
                process.send_signal(signal.SIGINT)
                returncode = process.wait(timeout=5)
            check(returncode == 130, f"constructor cancellation exit {returncode}")
            stdout = (self.output / f"{name}.stdout").read_bytes()
            stderr = (self.output / f"{name}.stderr").read_bytes()
            check(b"end=cancelled" in stdout, "constructor cancellation status")
            check(secret.encode() not in stdout + stderr and b"http://" not in stdout + stderr,
                  "constructor cancellation leaked source credentials")
            check(not mot.exists() and not raw.exists(), "constructor cancellation created outputs")
            self.results.append({"case": name, "returncode": returncode,
                                 "elapsed_s": time.monotonic() - start})
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            release.set()
            listener.close()
            server.join(timeout=6)
            check(not server.is_alive(), "constructor cancellation test server failed to stop")

    def concurrent_outputs(self, video, expected_prefix):
        """Exactly one concurrent creator owns a shared pair of new outputs."""
        mot, raw = self.output / "concurrent.txt", self.output / "concurrent.bin"
        command = self.command(video, mot, "--frames", "3", "--packets", raw)
        processes = []
        starts = []
        with ExitStack() as resources:
            try:
                for index in range(2):
                    stdout = resources.enter_context((self.output / f"concurrent-{index}.stdout").open("wb"))
                    stderr = resources.enter_context((self.output / f"concurrent-{index}.stderr").open("wb"))
                    starts.append(time.monotonic())
                    processes.append(subprocess.Popen(command, stdout=stdout, stderr=stderr))
                returncodes = [process.wait(timeout=120) for process in processes]
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
        check(sorted(returncodes) == [0, 1], f"concurrent creators must have one winner: {returncodes}")
        loser = returncodes.index(1)
        check(b"exists" in (self.output / f"concurrent-{loser}.stderr").read_bytes(),
              "concurrent loser must fail from output collision")
        actual, observations = packets(raw, 3)
        check(actual == expected_prefix, "concurrent output winner's packet prefix was changed")
        check_mot(mot, observations)
        for index, returncode in enumerate(returncodes):
            self.results.append({"case": f"concurrent-output-{index}", "returncode": returncode,
                                 "elapsed_s": time.monotonic() - starts[index]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track_stream", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    executable, engine, sequence = args.track_stream.resolve(), args.engine.resolve(), args.sequence.resolve()
    output_root = (args.output_root or executable.parent / "cli-tests").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=output_root))
    print(f"CLI artifacts: {output}", flush=True)
    harness = Harness(executable, engine, output)
    for index in range(1, 10):
        check((sequence / "img1" / f"{index:06d}.jpg").is_file(), "first nine JPEGs required")
    video = output / "first-nine.mkv"
    harness.run("make-video", [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                               "-framerate", "10", "-start_number", "1", "-i",
                               str(sequence / "img1" / "%06d.jpg"), "-frames:v", "9",
                               "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(video)])
    help_result = harness.run("help", [str(executable), "--help"])
    check(b"usage: track_stream" in help_result.stdout, "help usage")
    harness.open_interrupted()

    graph_mot, graph_raw = output / "graph.txt", output / "graph.bin"
    graph = harness.run("graph", harness.command(video, graph_mot, "--packets", graph_raw))
    check(b"output_profile=stream-observations/1" in graph.stdout, "online profile identity")
    check(b"backend=HOST_RGB_UPLOAD" in graph.stdout, "explicit CPU input backend")
    check(b"end=eof" in graph.stdout, "EOF status")
    graph_bytes, graph_observations = packets(graph_raw, 9)
    check_mot(graph_mot, graph_observations)

    serial_mot, serial_raw = output / "serial.txt", output / "serial.bin"
    harness.run("ordinary-serial", harness.command(video, serial_mot, "--serial", "--no-cuda-graph",
                                                   "--packets", serial_raw))
    serial_bytes, observations = packets(serial_raw, 9)
    check_mot(serial_mot, observations)
    check(serial_bytes == graph_bytes, "graph/ordinary serial complete packet bytes differ")
    check(serial_mot.read_bytes() == graph_mot.read_bytes(), "graph/ordinary serial MOT bytes differ")

    prefix_mot, prefix_raw = output / "prefix.txt", output / "prefix.bin"
    prefix = harness.run("frame-limit", harness.command(video, prefix_mot, "--frames", "3", "--packets", prefix_raw))
    prefix_bytes, observations = packets(prefix_raw, 3)
    check_mot(prefix_mot, observations)
    check(prefix_bytes == graph_bytes[:3 * PACKET_BYTES], "frame-limit packet prefix")
    check(b"end=frame-limit" in prefix.stdout, "frame-limit status")
    harness.concurrent_outputs(video, graph_bytes[:3 * PACKET_BYTES])

    protected = {path: digest(path) for path in (engine, video, graph_mot, graph_raw)}
    absent_mot = output / "must-not-create.txt"
    harness.run("existing-mot", harness.command(video, graph_mot), expected=1)
    harness.run("existing-packets", harness.command(video, absent_mot, "--packets", graph_raw), expected=1)
    harness.run("source-output-collision", harness.command(video, video), expected=1)
    harness.run("engine-output-collision", harness.command(video, engine), expected=1)
    harness.run("output-packet-collision", harness.command(video, absent_mot, "--packets", absent_mot), expected=1)
    harness.run("source-packet-collision", harness.command(video, absent_mot, "--packets", video), expected=1)
    check(all(digest(path) == original for path, original in protected.items()), "protected file changed")
    check(not absent_mot.exists(), "invalid output arguments created a file")
    harness.run("missing-source", harness.command(output / "missing.mkv", absent_mot), expected=1)
    harness.run("invalid-frame-limit", harness.command(video, absent_mot, "--frames", "0"), expected=1)
    harness.run("invalid-decode", harness.command(video, absent_mot, "--decode", "automatic"), expected=1)
    harness.run("invalid-device", harness.command(video, absent_mot, "--device", "-1"), expected=1)
    check(not absent_mot.exists(), "failed source/option created output")

    secret = "test-only-credential-never-log"
    privacy = harness.run("credential-redaction", harness.command(
        f"rtsp://tester:{secret}@127.0.0.1:1/video", absent_mot, "--timeout-ms", "1000"), expected=1, timeout=10)
    check(secret.encode() not in privacy.stdout + privacy.stderr and b"rtsp://" not in privacy.stdout + privacy.stderr,
          "source URL credentials leaked to CLI diagnostics")
    harness.interrupted(signal.SIGINT)
    harness.interrupted(signal.SIGTERM)

    unsupported_mot, unsupported_raw = output / "unsupported-nvdec.txt", output / "unsupported-nvdec.bin"
    unsupported = harness.run("nvdec-no-fallback", harness.command(
        video, unsupported_mot, "--packets", unsupported_raw, decode=None), expected=1)
    check(b"HOST_RGB_UPLOAD" not in unsupported.stdout, "NVDEC failure used CPU preprocessing")
    check(not unsupported_mot.exists() and not unsupported_raw.exists(), "unsupported GPU codec created outputs")

    nvdec_cases = []
    for name, encoder, pixel_format in (
            ("h264-8", "libx264", "yuv420p"),
            ("hevc-8", "libx265", "yuv420p"),
            ("hevc-10", "libx265", "yuv420p10le")):
        encoded = output / f"{name}.mkv"
        codec_options = ["-preset", "fast", "-crf", "20"]
        if encoder == "libx264":
            codec_options += ["-bf", "2", "-g", "9"]
        else:
            codec_options += ["-x265-params", "bframes=2:keyint=9:min-keyint=9:log-level=error:pools=2:frame-threads=1"]
        # JPEG frame properties can override encoder-only color tags. Convert
        # the full-range JPEG matrix, then set frame metadata explicitly so the
        # actual bitstream carries the SDR transfer/primaries being tested.
        color_filter = ("scale=in_color_matrix=bt601:out_color_matrix=bt709:in_range=full:out_range=limited,"
                        f"format={pixel_format},"
                        "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709")
        harness.run(f"make-{name}", [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
                    "-framerate", "10", "-start_number", "1", "-i",
                    str(sequence / "img1" / "%06d.jpg"), "-frames:v", "9", "-vf", color_filter, "-c:v", encoder,
                    *codec_options, "-pix_fmt", pixel_format,
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-color_range", "tv", str(encoded)])
        probe = harness.run(f"probe-{name}", [args.ffprobe, "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name,pix_fmt,color_space,color_transfer,color_range",
                            "-show_entries", "frame=pict_type", "-of", "json", str(encoded)])
        metadata = json.loads(probe.stdout)
        stream = metadata["streams"][0]
        check(stream["codec_name"] == ("h264" if encoder == "libx264" else "hevc"), "encoded codec")
        check(stream["pix_fmt"] == pixel_format, "encoded bit depth/pixel layout")
        check(stream["color_space"] == stream["color_transfer"] == "bt709" and stream["color_range"] == "tv",
              "SDR limited-range color contract")
        check(len(metadata["frames"]) == 9 and any(frame["pict_type"] == "B" for frame in metadata["frames"]),
              "NVDEC fixture must exercise B-frame presentation reorder and decoder drain")

        gpu_mot, gpu_raw = output / f"{name}-graph.txt", output / f"{name}-graph.bin"
        # Omit --decode: this checks the user-facing CUDA default itself.
        gpu_graph = harness.run(f"nvdec-{name}-graph", harness.command(
            encoded, gpu_mot, "--packets", gpu_raw, "--device", "0", decode=None))
        check(b"video.backend=ffmpeg-nvdec" in gpu_graph.stdout, "default decode must use NVDEC")
        check(b"backend=CUDA_DEVICE_VIDEO" in gpu_graph.stdout, "NVDEC preprocessing must use device input")
        check(b"HOST_RGB_UPLOAD" not in gpu_graph.stdout and b"rgb_download" not in gpu_graph.stdout,
              "GPU decode identity must not claim a host RGB path")
        check(b"end=eof" in gpu_graph.stdout, "NVDEC decoder EOF drain")
        gpu_bytes, observations = packets(gpu_raw, 9)
        check(bool(observations), "real-image NVDEC comparison must include tracked rows")
        check_mot(gpu_mot, observations)
        ordinary_mot, ordinary_raw = output / f"{name}-serial.txt", output / f"{name}-serial.bin"
        gpu_serial = harness.run(f"nvdec-{name}-serial", harness.command(
            encoded, ordinary_mot, "--packets", ordinary_raw, "--serial", "--no-cuda-graph", decode="cuda"))
        check(b"video.backend=ffmpeg-nvdec" in gpu_serial.stdout, "explicit CUDA decode")
        ordinary_bytes, observations = packets(ordinary_raw, 9)
        check_mot(ordinary_mot, observations)
        check(ordinary_bytes == gpu_bytes, f"{name}: graph/serial full packets differ")
        check(ordinary_mot.read_bytes() == gpu_mot.read_bytes(), f"{name}: graph/serial MOT bytes differ")
        if name == "h264-8":
            gpu_prefix_mot, gpu_prefix_raw = output / "nvdec-prefix.txt", output / "nvdec-prefix.bin"
            prefix_result = harness.run("nvdec-frame-limit", harness.command(
                encoded, gpu_prefix_mot, "--packets", gpu_prefix_raw, "--frames", "3", decode=None))
            check(b"end=frame-limit" in prefix_result.stdout, "NVDEC frame-limit status")
            prefix_bytes, observations = packets(gpu_prefix_raw, 3)
            check_mot(gpu_prefix_mot, observations)
            check(prefix_bytes == gpu_bytes[:3 * PACKET_BYTES], "NVDEC frame-limit packet prefix")
        nvdec_cases.append({"codec": name, "video_sha256": digest(encoded),
                            "packets": 9, "graph_serial_packet_bytes_equal": True,
                            "graph_serial_mot_bytes_equal": True})

    report = {"formal_acceptance": False, "scope": "nine-frame CPU/NVDEC local development CLI integration",
              "binary_sha256": digest(executable), "engine_sha256": protected[engine],
              "input_sha256": digest(video), "cases": harness.results,
              "graph_serial_packet_bytes_equal": True, "graph_serial_mot_bytes_equal": True,
              "nvdec": nvdec_cases, "nvdec_unsupported_codec_rejected_without_fallback": True}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(harness.results)} CLI cases; artifacts: {output}")


if __name__ == "__main__":
    main()
