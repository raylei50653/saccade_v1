#!/usr/bin/env python3
"""Optional loopback-only server for the offline replay, with no dependencies."""
import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class ReplayHandler(SimpleHTTPRequestHandler):
    """Support byte ranges so browsers can seek without downloading the movie."""

    def end_headers(self):
        if self.path.split('?', 1)[0].endswith('.mp4'):
            self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def send_head(self):
        self.byte_range = None
        requested = self.headers.get('Range')
        path = Path(self.translate_path(self.path))
        if not requested or path.suffix.lower() != '.mp4' or not path.is_file():
            return super().send_head()
        stream = path.open('rb')
        size = os.fstat(stream.fileno()).st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested.strip())
        valid = bool(match and any(match.groups()) and size > 0)
        start, end = 0, size - 1
        if valid:
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size - 1) if last else size - 1
                valid = start <= end and start < size
            else:
                suffix = int(last)
                valid = suffix > 0
                start = max(0, size - suffix)
        if not valid:
            stream.close()
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        self.byte_range = (start, end)
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(end - start + 1))
        self.end_headers()
        stream.seek(start)
        return stream

    def copyfile(self, source, output):
        try:
            if self.byte_range is None:
                return super().copyfile(source, output)
            remaining = self.byte_range[1] - self.byte_range[0] + 1
            while remaining:
                block = source.read(min(256 * 1024, remaining))
                if not block:
                    break
                output.write(block)
                remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers cancel old requests when a new frame is sought.
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    with ThreadingHTTPServer(('127.0.0.1', args.port),
                             partial(ReplayHandler, directory=str(directory))) as server:
        print(f'Saccade v1 demo: http://127.0.0.1:{server.server_port}/', flush=True)
        print('Press Ctrl-C to stop.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
