import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('0.0.0.0', 25565))  # Listen on all interfaces
s.listen(1)
print("Listening on port 25565")
conn, addr = s.accept()
print(f"Connection received from {addr}")


