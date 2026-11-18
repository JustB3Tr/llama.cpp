#!/usr/bin/env python3
"""Run a local llama-server experiment; keep raw server logs and per-request counters."""
import argparse
import json
import os
import pathlib
import socket
import subprocess
import time
import urllib.error
import urllib.request


def request(url, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def proc_io(pid):
    try:
        return {k: int(v) for k, v in (line.split(":") for line in pathlib.Path(f"/proc/{pid}/io").read_text().splitlines())}
    except (OSError, ValueError):
        return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--server', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True, help='new directory for logs and measurements')
    p.add_argument('--port', type=int, default=18989)
    p.add_argument('--tokens', type=int, default=16)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--startup-timeout', type=int, default=900)
    p.add_argument('--request-timeout', type=int, default=900)
    p.add_argument('flags', nargs=argparse.REMAINDER, help='server flags after --')
    a = p.parse_args()
    if a.tokens < 1 or a.repeats < 1:
        p.error('tokens and repeats must be positive')
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', a.port))
        except OSError:
            p.error('port is already in use; choose a free --port')
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    flags = a.flags[1:] if a.flags[:1] == ['--'] else a.flags
    cmd = [a.server, '-m', a.model, '--host', '127.0.0.1', '--port', str(a.port)] + flags
    result = {'command': cmd, 'started_unix': time.time(), 'requests': [],
              'note': 'read_bytes is process-attributed Linux I/O, not necessarily physical device traffic; cache state is uncontrolled'}
    server = None
    try:
        with (out / 'server.log').open('w', encoding='utf-8') as log:
            server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + a.startup_timeout
            url = f'http://127.0.0.1:{a.port}'
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'server exited during startup: {server.returncode}')
                if time.monotonic() > deadline:
                    raise TimeoutError('server startup timed out')
                try:
                    if request(url + '/health').get('status') == 'ok':
                        break
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                time.sleep(1)
            result['startup_seconds'] = time.time() - result['started_unix']
            for _ in range(a.repeats):
                before = proc_io(server.pid)
                start = time.monotonic()
                response = request(url + '/completion', {'prompt': 'Explain why the sky is blue in a few sentences.',
                    'n_predict': a.tokens, 'temperature': 0, 'seed': 42, 'cache_prompt': False}, a.request_timeout)
                elapsed = time.monotonic() - start
                after = proc_io(server.pid)
                result['requests'].append({'elapsed_seconds': elapsed, 'timings': response.get('timings'),
                    'tokens_predicted': response.get('tokens_predicted'), 'stopped_limit': response.get('stopped_limit'),
                    'process_read_bytes': after.get('read_bytes', 0) - before.get('read_bytes', 0) if before and after else None,
                    'content': response.get('content')})
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        result['server_exit'] = server.returncode if server else None
        (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if 'error' in result:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
