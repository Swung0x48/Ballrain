import socket
from Message import MsgType, MsgGameState
from Message import msg_body_len

class TCPServer:
    def __init__(self, host='127.0.0.1', port=27787):
        self.client_socket = None
        self.client_address = None
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.msg_size = {

        }

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.running = True

            print(f"Waiting connection at {self.host}:{self.port}")
            self.client_socket, self.client_address = self.socket.accept()
            print(f"Accepted client at: {self.client_address}")
        except Exception as e:
            print(f"Server error: {e}")

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
        print("Server stopped")

    def recv(self, length):
        data = self.client_socket.recv(length)
        return data

    def send(self, data):
        self.client_socket.send(data)

    def recv_msg(self):
        btype = self.recv(4)
        msg_type = int.from_bytes(btype, byteorder='little')
        msg_body = self._recv_msg_body(msg_type)
        return msg_type, msg_body

    def _recv_msg_body(self, msg_type):
        if msg_body_len[msg_type] > 0:
            bmsg_body = self.recv(msg_body_len[msg_type])
            return MsgGameState(bmsg_body)
        else:
            return None

    def send_msg(self, msg_type: MsgType, msg_body):
        bmsg_type = msg_type.value.to_bytes(4, byteorder='little')
        self.send(bmsg_type)
        if msg_body is not None:
            self.send(msg_body)

if __name__ == "__main__":
    server = TCPServer()
    try:
        server.start()
        while True:
            b = server.recv(4)
            if b:
                val = int.from_bytes(b, byteorder='little')
                print(f'rcvd: {val}')
                val += 1
                out_bytes = val.to_bytes(length=4, byteorder='little')
                server.send(out_bytes)
                print(f'sent: {val}')
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()