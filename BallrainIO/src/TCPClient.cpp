#include "TCPClient.h"
#include <cassert>


TCPClient::TCPClient(): m_recvBuffer(512)
{
    if (WSAStartup(MAKEWORD(2, 2), &m_wsaData) != 0) {
        m_lastError = WSAGetLastError();
    }
}

TCPClient::~TCPClient()
{
    WSACleanup();
}

int TCPClient::GetLastError()
{
    int err = m_lastError;
    m_lastError = 0;
    return err;
}

bool TCPClient::Connect(std::string host, int port)
{
    m_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (m_socket == INVALID_SOCKET) {
        return false;
    }

    m_serverAddr.sin_family = AF_INET;
    m_serverAddr.sin_port = htons(port);

    if (inet_pton(AF_INET, host.c_str(), &m_serverAddr.sin_addr) <= 0) {
        // if inet_pton failed, try getaddrinfo to resolve address
        struct addrinfo hints, * result;
        ZeroMemory(&hints, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;

        if (getaddrinfo(host.c_str(), NULL, &hints, &result) == 0) {
            m_serverAddr.sin_addr = ((struct sockaddr_in*)result->ai_addr)->sin_addr;
            freeaddrinfo(result);
        }
        else {
            closesocket(m_socket);
            return false;
        }
    }

    if (connect(m_socket, (sockaddr*)&m_serverAddr, sizeof(m_serverAddr)) == SOCKET_ERROR) {
        closesocket(m_socket);
        return false;
    }

    m_connected = true;
	return true;
}

void TCPClient::Disconnect()
{
    if (m_socket != INVALID_SOCKET) {
        closesocket(m_socket);
        m_socket = INVALID_SOCKET;
    }
    m_connected = false;
}

int TCPClient::Send(const void* data, int len) {
    if (!m_connected)
        return SOCKET_ERROR;

    int bytesSent = send(m_socket, (const char*)data, len, 0);
    if (bytesSent == SOCKET_ERROR) {
        m_connected = false;
    }
    return bytesSent;
}

int TCPClient::Receive(int len, void* dest)
{
    if (!m_connected)
        return SOCKET_ERROR;

    char* ptr = (char*)dest;
    int bytesReceived = 0;
    while (bytesReceived < len) {
        int rcvd = recv(m_socket, ptr + bytesReceived, len - bytesReceived, 0);
        if (bytesReceived < 0) {
            m_connected = false;
            break;
        }
        bytesReceived += rcvd;
    }
    
    return bytesReceived;
}

int TCPClient::SendMsg(MessageType type, const void* data)
{
    auto sz = Send(&type, sizeof(type));
    switch (type) {
        case MessageType::BRM_GameState: {
            assert(data != nullptr);
            auto datasz = Send(data, sizeof(MsgGameState));
            if (datasz < 0)
                return datasz;
            else
                sz += datasz;
            break;
        }
        case MessageType::BRM_MsgSceneRep: {
            auto* msg = (MsgSceneRep*)data;
            int count = msg->floorBoxes.size();
            auto datasz = Send(&count, sizeof(int));
            if (datasz < 0)
                return datasz;
            for (int i = 0; i < count; ++i) {
                auto* box = &msg->floorBoxes[i];
                auto sentsz = Send(box, sizeof(VxBbox));
                if (sentsz < 0)
                    return sentsz;
                else
                    datasz += sentsz;
            }
            sz += datasz;
        }
    }
    return sz;
}

MessageType TCPClient::ReceiveMsg()
{
    MessageType type = MessageType::BRM_InvalidType;
    auto sz = Receive(sizeof(MessageType), &type);
    switch (type) {
        case MessageType::BRM_KbdInput: {
            m_recvBuffer.resize(sizeof(MsgKbdInput));
            auto msgsz = Receive(sizeof(MsgKbdInput), m_recvBuffer.data());
            if (msgsz < 0)
                return MessageType::BRM_InvalidType;
        }
            
    }
    return type;
}

const std::vector<uint8_t>& TCPClient::GetMessageBuf() const
{
    return m_recvBuffer;
}


