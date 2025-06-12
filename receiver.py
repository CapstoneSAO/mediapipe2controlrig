import socket
import struct
import json
import threading
import asyncio

class Receiver:
    def __init__(self, host, port, queue, loop):
        self.host = host
        self.port = port
        self.queue = queue
        self._thread = None
        self._stop_event = threading.Event()
        self.loop = loop
        self._sock = None  # Socket reference for cleanup
        self._conn = None  # Connection reference for cleanup

    def start_in_thread(self):
        self._thread = threading.Thread(
            target=self.listening,
            args=(self.host, self.port, self.queue, self._stop_event, self.loop, self),
            daemon=True
        )
        self._thread.start()
        print("[Receiver] Listening thread started")

    @staticmethod
    def listening(host, port, queue, stop_event, loop, self_ref):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self_ref._sock = sock
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(1)
        conn, addr = sock.accept()
        print(f"[Receiver]: Connected to unreal server at {host}:{port}")
        self_ref._conn = conn
        with conn:
            while not stop_event.is_set():
                hdr = Receiver.recv_exact(conn, 4)
                print("[Receiver] Waiting for header...")
                if hdr is None:
                    print("[Python] Unreal 斷線")
                    break
                body_len = struct.unpack(">I", hdr)[0]
                body = Receiver.recv_exact(conn, body_len)
                if body is None:
                    print("[Python] Unreal 斷線")
                    break
                try:
                    json_str = body.decode("utf-8").strip('\x00\r\n\t ')
                    obj = json.loads(json_str)
                    asyncio.run_coroutine_threadsafe(queue.put(obj), loop)
                    print("[Receiver] Received object:", obj)
                except Exception as e:
                    print("[Python] JSON 解析失敗：", e)
                    continue

    @staticmethod
    def recv_exact(sock, nbytes):
        buf = bytearray()
        while len(buf) < nbytes:
            chunk = sock.recv(nbytes - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def close(self):
        print("[Receiver] Closing resources...")
        if self._stop_event:
            self._stop_event.set()
        if self._conn:
            try:
                self._conn.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._conn.close()
                print("[Receiver] Connection closed.")
            except Exception:
                pass
        if self._sock:
            try:
                self._sock.close()
                print("[Receiver] Socket closed.")
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join()
        print("[Receiver] Closed.")