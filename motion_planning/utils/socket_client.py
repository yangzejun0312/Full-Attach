import socket
import threading

class SocketClientBase:
    def __init__(self, host='127.0.0.1', port=5000, buffer_size=1024):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.running = False

    def connect(self):
        self.client_socket.connect((self.host, self.port))
        self.running = True
        print(f"[Client] Connected to {self.host}:{self.port}")
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def _listen_loop(self):
        while self.running:
            try:
                data = self.client_socket.recv(self.buffer_size)
                if not data:
                    break
                self.on_message(data.decode())
            except ConnectionResetError:
                break
        self.client_socket.close()

    def on_message(self, message):
        """Override this method in subclasses to handle received messages."""
        print(f"[Client] Received: {message}")

    def send(self, message):
        self.client_socket.sendall(message.encode())

    def disconnect(self):
        self.running = False
        self.client_socket.close()
        print("[Client] Disconnected")