#!/usr/bin/env python3
"""
Web Teleop Relay Server
=======================
Flask + Flask-SocketIO server that:
  1. Serves the front-end HTML page.
  2. Receives teleop_cmd JSON via WebSocket from the browser.
  3. Packs the data into a C-struct binary (UdpCommand) via UDP to the C++ node.

Usage:
    python app.py [--udp-port 9870] [--web-port 5000]

UdpCommand C-struct layout (17 bytes, packed):
    int32_t  mode     (4 bytes)
    float    vx       (4 bytes)  ratio [-1, 1]
    float    vy       (4 bytes)  ratio [-1, 1]
    float    yaw      (4 bytes)  ratio [-1, 1]
    uint8_t  e_stop   (1 byte)   0 or 1
"""

import argparse
import socket
import struct
import os

from flask import Flask, send_from_directory
from flask_socketio import SocketIO

# ---- Parse CLI arguments ----
parser = argparse.ArgumentParser(description='Web Teleop Relay Server')
parser.add_argument('--udp-port', type=int, default=9870,
                    help='UDP port to send commands to C++ node (default: 9870)')
parser.add_argument('--web-port', type=int, default=5000,
                    help='Web server port (default: 5000)')
args = parser.parse_args()

UDP_HOST = '127.0.0.1'
UDP_PORT = args.udp_port
WEB_PORT = args.web_port

# ---- Flask app ----
app = Flask(__name__, static_folder='.', static_url_path='')
app.config['SECRET_KEY'] = 'quadruped-teleop'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# ---- UDP socket (non-blocking, fire-and-forget) ----
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ---- C-struct pack format ----
# '<' = little-endian
# 'i'  = int32_t  (mode)
# '3f' = 3 × float (vx, vy, yaw)
# 'B'  = uint8_t  (e_stop)
PACK_FMT = '<i3fB'
PACK_SIZE = struct.calcsize(PACK_FMT)  # 17 bytes


@app.route('/')
def index():
    """Serve the front-end HTML page."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)),
                               'index.html')


@socketio.on('teleop_cmd')
def handle_teleop_cmd(data):
    """
    Receive teleop command from browser, pack and forward via UDP.

    Expected JSON:
        {
            "mode": 0-3,
            "vx_ratio": float [-1, 1],
            "vy_ratio": float [-1, 1],
            "yaw_ratio": float [-1, 1],
            "e_stop": bool
        }
    """
    try:
        mode     = int(data.get('mode', 0))
        vx       = float(data.get('vx_ratio', 0.0))
        vy       = float(data.get('vy_ratio', 0.0))
        yaw      = float(data.get('yaw_ratio', 0.0))
        e_stop   = 1 if data.get('e_stop', False) else 0

        packet = struct.pack(PACK_FMT, mode, vx, vy, yaw, e_stop)
        udp_sock.sendto(packet, (UDP_HOST, UDP_PORT))
    except Exception as e:
        print(f'[teleop] Error packing/sending: {e}')


@socketio.on('connect')
def handle_connect():
    print('[teleop] Browser connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('[teleop] Browser disconnected')


if __name__ == '__main__':
    print(f'[teleop] UDP target: {UDP_HOST}:{UDP_PORT}  '
          f'(pack size: {PACK_SIZE} bytes)')
    print(f'[teleop] Web server: http://0.0.0.0:{WEB_PORT}')
    socketio.run(app, host='0.0.0.0', port=WEB_PORT, debug=False)
