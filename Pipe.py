import win32pipe
import win32file


class Pipe:
    def __init__(self, name):
        self.name = name
        self.pipe_handle = None
        print("Opened BallancePipe. Waiting for connection...")

    def run(self):
        self.pipe_handle = win32pipe.CreateNamedPipe(
            rf"\\.\pipe\{self.name}",
            win32pipe.PIPE_ACCESS_DUPLEX,  # Duplex communication
            win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
            1,
            65536,  # Out buffer size
            65536,  # In buffer size
            0,  # Default timeout
            None  # Default security attributes
        )
        print(f"Pipe server created: {self.name}")

        print("Waiting for client...")
        win32pipe.ConnectNamedPipe(self.pipe_handle, None)
        print("got client")

    def _read_raw(self, size):
        resp = win32file.ReadFile(self.pipe_handle, size)
        return resp

    def _write_raw(self, data):
        win32file.WriteFile(self.pipe_handle, data)