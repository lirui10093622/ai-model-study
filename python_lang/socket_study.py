import socket_study

s = socket.socket()
s.bind(('localhost', 8080))
print('Waiting for connection...')
s.listen()

while True:
    conn, addr = s.accept()
    print('Connected by', addr)
    while True:
        data = conn.recv(1024)
        if not data:
            break
        conn.sendall(data)