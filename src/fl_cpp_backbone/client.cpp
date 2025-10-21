//  Created by Ioannis Christofilogiannis on 17/1/24.

#include "net_lib.h"
#include "shared_buffer.h"
#include "barrier.h"



// A TCP client that will send data to a server and then receive new data
class FL_Client {
private:
    int sock = 0;
    struct sockaddr_in serv_addr;
    SharedBuffer* aBuffer;
    SharedBuffer* pBuffer;

public:
    // Thread safe tools
    std::mutex mtx;
    std::mutex params_mtx;
    std::condition_variable cond_var_client;
    //std::atomic<bool> aggregationProcessed;
    std::atomic<bool> paramsReady;
    std::atomic<bool> stop_rounds;
    int clientID = 0;

    int getMachineID() {
        std::cout << std::to_string(clientID) << std::endl;
        return clientID;
    }

    // Notify the thread to wake up, called externally by Python
    void setParamsReady(bool s_rounds) {
//        std::cout << "s_rounds value:" << std::to_string(s_rounds) << std::endl;
        if (s_rounds){
            stop_rounds.store(true, std::memory_order_release);
        } else {
            stop_rounds.store(false, std::memory_order_release);
        }
        paramsReady.store(true, std::memory_order_release);
        //paramsReady = true;
//        stop_rounds = s_rounds;
        cond_var_client.notify_one();
    }


    // Client constructor, setup the connection
    FL_Client(const char* server_ip, int server_port, SharedBuffer* paramsBuffer, SharedBuffer* aggregationBuffer) {
        aBuffer = aggregationBuffer;
        pBuffer = paramsBuffer;
        stop_rounds.store(false, std::memory_order_release);
        paramsReady.store(false, std::memory_order_release);

#ifdef _WIN32
        // Initialize Winsock for Windows
        WSADATA wsaData;
        int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (result != 0) {
            printError(-1, "WSAStartup failed");
        }
#endif

        // Creating socket file descriptor
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock == -1) {
            printError(sock, "Socket creation error");
        }
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(server_port);
        
        // Convert IPv4 and IPv6 addresses from text to binary form
        if(inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
           printError(sock, "Invalid address/ Address not supported");
        }

        // Connection attempt - master server
        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
            printError(sock, "Connection Failed");
        }

        int client_mid;
        recv(sock, (char*)&client_mid, sizeof(client_mid), 0);
        clientID = client_mid;

    }

    // Participates in the Federated Learning experiment as a client
    // Uses detached thread to continue Python operations
    void participate() {
         std::thread clientThread(&FL_Client::participateThread, this);
         clientThread.detach();
    }

    // Thread that handles the communication loop
    void participateThread() {
        int count = 1;
        while (true) {
            // Initialize scope vars
            char ackBuffer[4] = {0};

            // Wait for Python update on whether to stop or not
            {
                std::unique_lock<std::mutex> lock(mtx);
                cond_var_client.wait(lock, [this]{ return paramsReady.load(); });
            }

             // Decide whether to stop the rounds based on the server message inside aggregated data
            if(stop_rounds.load() == true){
                break;
            }
            // Reset params ready condition
            //paramsReady.store(false, std::memory_order_release);


            //------------------ 1. Sending Params ------------------//
            std::cout << "\nClient "+std::to_string(clientID)+" begun training round: " << std::to_string(count) << std::endl;
            // Get params buffer size and data md5 checksum
            int buffSize = (*pBuffer).size();
            std::cout << "\nClient pbuffer size: "+std::to_string(buffSize) << std::endl;
            std::string paramsMD5 = (*pBuffer).md5_str;

            // Create header structure
            std::string header = setupHeader(paramsMD5, clientID, buffSize);
            // Send the header structure we created
            send(sock, header.c_str(), header.size(), 0);
            std::cout << "Header Sent: " << header;

            // Wait for header ACK in order to send the file
            recv(sock, ackBuffer, sizeof(ackBuffer), 0);

            std::cout << "Received header ACK from server, sending file: " << std::endl;

            // Read params buffer
            char* pData = new char[buffSize];
            (*pBuffer).read(pData, buffSize, 0);
            // Clear pbuffer
            pBuffer->clear();
            sendBufferSock(sock, pData, buffSize);
            delete[] pData;

            // Wait for file ack
            char ackBuffer2[4] = {0};
            recv(sock, ackBuffer2, sizeof(ackBuffer2), 0);

            std::cout << "Received file ACK from server, File sent successfully. " << std::endl;

            //------------------- 2. Aggregation -------------------//
            // Receive aggregation data header from server
            std::string receivedMD5, machineID;
            int64_t fileSize;
            std::string headerString = receiveHeaderSTR(sock);
            parseHeader(headerString, receivedMD5, machineID, fileSize);
            std::cout << "Aggregated data header: "+ headerString;

            // Send header ack to server
            send(sock, ackMessage, strlen(ackMessage), 0);
            // Create temp char* buffer to receive bytes
            char* sData = new char[fileSize];
            // Receive data from server to temp buffer
            receiveBufferSock(sock, sData, fileSize);
            // Store data from temp buffer to appropriate client buffer and set data as complete
            aBuffer->write(sData, fileSize);
            // Sets flag read by readParams in Python
            aBuffer->setComplete();
            // Clear temp data
            delete[] sData;
            std::cout << "Received aggregated data from server, sending ACK. " << std::endl;
            // Send final ACK
            send(sock, ackMessage, strlen(ackMessage), 0);
            // Increment aggregation rounds
            count++;
            paramsReady.store(false, std::memory_order_release);
        }
        std::cout << "Exited the matrix" << std::endl;
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return;
    }

    // Wrapper method used to make parse header method available to Python
    void parseHeaderWrapper(const std::string& header, std::string& receivedMD5, std::string& machineID, int64_t& fileSize) {
        ::parseHeader(header, receivedMD5, machineID, fileSize);
    }
    
    // Destructor - for garbage collection
    ~FL_Client() {
#ifdef _WIN32
        closesocket(sock);
        WSACleanup();
#else
        close(sock);
#endif
    }
};