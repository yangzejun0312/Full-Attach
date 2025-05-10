import socket
import threading

class SocketServerBase:
    def __init__(self, host='127.0.0.1', port=5000, buffer_size=1024):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_sockets = []
        self.running = False

    def start(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        self.running = True
        print(f"[Server] Listening on {self.host}:{self.port}")
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while self.running:
            if len(self.client_sockets) == 0:
                client_sock, addr = self.server_socket.accept()
                self.client_sockets.append(client_sock)
                print(f"[Server] Connection from {addr}")
                threading.Thread(target=self._handle_client, args=(client_sock,), daemon=True).start()
            else:
                pass

    def _handle_client(self, client_sock):
        while self.running:
            try:
                data = client_sock.recv(self.buffer_size)
                if not data:
                    break
                self.on_message(data.decode(), client_sock)
            except ConnectionResetError:
                break
        client_sock.close()
        self.client_sockets.remove(client_sock)

    def on_message(self, message, client_sock):
        """Override this method in subclasses to handle received messages."""
        print(f"[Server] Received: {message}")

    def send(self, message, client_sock):
        client_sock.sendall(message.encode())

    def broadcast(self, message):
        for client in self.client_sockets:
            self.send(message, client)

    def stop(self):
        self.running = False
        for client in self.client_sockets:
            client.close()
        self.client_sockets.clear()
        self.server_socket.close()
        print("[Server] Shutdown")