import socket

from utils import PoseObject

from loguru import logger

def create_unreal_sender_socket(host='127.0.0.1', port=5001, wait_for_connection=True):
    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sender_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sender_socket.bind((host, port))
    sender_socket.listen(1)
    logger.info(f"Socket created on {host}:{port}")
    if wait_for_connection:
        logger.info("Waiting for connection...")
        conn, addr = sender_socket.accept()
        logger.info(f"Socket {conn.getsockname()} connected by {addr}")
        return conn
    else:
        logger.info("Socket created but not waiting for connection.")
    return sender_socket


def wait_for_socket_connection(_socket: socket.socket, blocking=True, timeout=5):
    """等待 socket 連線"""
    if blocking:
        while True:
            try:
                conn, addr = _socket.accept()
                print(f"Socket {_socket.getsockname()} connected by {addr}")
                return conn
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                _socket.close()
                return None
    else:
        _socket.settimeout(timeout)
        try:
            conn, addr = _socket.accept()
            print(f"Socket {_socket.getsockname()} connected by {addr}")
            return conn
        except socket.timeout:
            print("Socket connection timed out.")
            return None
        except KeyboardInterrupt:
            _socket.close()
            return None


def send_data_as_packet_to_unreal(conn: socket.socket, data: str):
    """將資料打包並傳送到 Unreal"""
    if not conn:
        print("No connection to send data.")
        return

    if not data:
        print("No data to send.")
        return

    try:
        # 將資料轉換為 PoseObject 並打包
        pose_object = PoseObject(data)
        packet = pose_object.packet
    except Exception as e:
        print(f"Error packaging data: {e}")

    try:
        # 傳送資料
        conn.sendall(packet)
    except Exception as e:
        print(f"Error sending data: {e}")
        conn.close()


def send_data_to_unreal(conn: socket.socket, data: str):
    """將資料傳送到 Unreal"""
    if not conn:
        print("No connection to send data.")
        return

    try:
        conn.sendall(data.encode('utf-8'))
        print(f"Sent data: {data}")
    except Exception as e:
        print(f"Error sending data: {e}")
        conn.close()


def close_unreal_socket(conn: socket.socket):
    """關閉 Unreal socket 連線"""
    if conn:
        try:
            conn.close()
            print("Unreal socket closed.")
        except Exception as e:
            print(f"Error closing socket: {e}")
    else:
        print("No connection to close.")
