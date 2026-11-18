import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import unittest

BENCH = pathlib.Path(__file__).resolve().parents[1] / 'tools' / 'lowram_bench.py'
MOCK = '''#!/usr/bin/env python3
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def reply(self, obj):
        data = json.dumps(obj).encode(); self.send_response(200); self.end_headers(); self.wfile.write(data)
    def do_GET(self): self.reply({'status': 'ok'})
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.reply({'content': 'ok', 'tokens_predicted': 2, 'timings': {'predicted_n': 2}})
HTTPServer(('127.0.0.1', int(sys.argv[sys.argv.index('--port')+1])), Handler).serve_forever()
'''


@unittest.skipIf(sys.platform == 'win32', 'executable mock uses a Unix shebang')
class BenchTests(unittest.TestCase):
    def test_server_lifecycle_and_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            server = root / 'server'
            server.write_text(MOCK)
            server.chmod(0o755)
            with socket.socket() as s:
                s.bind(('127.0.0.1', 0))
                port = s.getsockname()[1]
            out = root / 'out'
            run = subprocess.run([sys.executable, str(BENCH), '--server', str(server), '--model', 'fake.gguf',
                                  '--out', str(out), '--port', str(port), '--tokens', '2', '--repeats', '2'],
                                 capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stderr)
            data = json.loads((out / 'result.json').read_text())
            self.assertEqual(len(data['requests']), 2)
            self.assertEqual(data['requests'][0]['tokens_predicted'], 2)
            self.assertTrue((out / 'server.log').exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            run = subprocess.run([sys.executable, str(BENCH), '--server', '/unused', '--model', '/unused', '--out', directory],
                                 capture_output=True, text=True, timeout=5)
            self.assertNotEqual(run.returncode, 0)


if __name__ == '__main__':
    unittest.main()
