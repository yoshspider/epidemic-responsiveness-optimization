import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from epidemic.monthly_download import download


class MonthlyDownloadTests(unittest.TestCase):
    def test_range_assembly_and_checkpoint_resume(self):
        payload = bytes(range(256)) * 4096

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Last-Modified', 'Wed, 14 Oct 2026 12:00:00 GMT')
                self.end_headers()

            def do_GET(self):
                first, last = map(int, self.headers['Range'].removeprefix('bytes=').split('-'))
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {first}-{last}/{len(payload)}')
                self.send_header('Content-Length', str(last - first + 1))
                self.end_headers()
                self.wfile.write(payload[first:last + 1])
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'archive.zip'
                url = f'http://127.0.0.1:{server.server_port}/archive'
                download(url, path, workers=3, block_size=65536)
                self.assertEqual(
                    hashlib.sha256(
                        path.read_bytes()).digest(),
                    hashlib.sha256(payload).digest())
                # Simulate interruption after range assembly, before final rename.
                path.replace(path.with_suffix('.ranged.part'))
                download(url, path, workers=3, block_size=65536)
                self.assertEqual(path.read_bytes(), payload)
        finally:
            server.shutdown()
            server.server_close()
