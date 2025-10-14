#ifndef BALLRAIN_TCPCLIENT_H
#define BALLRAIN_TCPCLIENT_H

#include <string>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>

#pragma comment(lib, "ws2_32.lib")

class TCPClient {
public:
	enum class MessageType: int {
		BRM_BallNavActive,
		BRM_BallNavInactive,
		BRM_BallState,
		BRM_InvalidType
	};

	TCPClient();
	~TCPClient();
	int GetLastError();
	bool Connect(std::string host, int port);
	void Disconnect();
	int Send(const void* data, int len);
	int Receive(int len, void* dest);
	int SendMsg(MessageType type, void* data = nullptr);
private:
	int m_lastError = 0;
	bool m_connected = false;
	WSADATA m_wsaData;
	SOCKET m_socket;
	sockaddr_in m_serverAddr;
};

#endif // !BALLRAIN_TCPCLIENT_H
