//  Created by Ioannis Christofilogiannis on 17/1/24.

// Include necessary data structures
#include "net_lib.h"
#include "shared_buffer.h"
#include "barrier.h"



// A Multi-threaded server that will receive data from each client and after the aggregation signal is received,
// will send the aggregated data formed with Python back to each client
class FL_Server {
private:
    // Server characteristics
    int server_fd, new_socket;
    int masterPort = 8080;
    struct sockaddr_in address;
    int opt = 1;
    int numOfClients;
    //const int MAX_NUM_OF_CLIENTS = 10;
    std::atomic<bool> aggregation;
    std::atomic<bool> stop_rounds;

    // Thread-Safe tools:
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<int> clientCount;
    std::atomic<int> completedJobs;

    // Keep a global barrier variable for thread synchronization
    std::unique_ptr<Barrier> globalBarrier;

    // Make a buffer for the aggregation data
    BufferManager* aggregationBuffer;

    // Make buffers for each clients received data
    BufferManager* clientBuffers;

    // Handle client connection, on a new spawned thread
    void handleClientConnection(int clientSocket, int cCount) {
        // Assign machine ID to client
        int client_id = cCount;
        send(clientSocket, (const char*)&client_id, sizeof(client_id), 0);
        int count = 0;
        while (true) {
            // Each client connection must have it's own ACK buffer
            char ackBuffer[4] = {0};

            // Receive header from client:
            // Initialize strings
            std::string headerString = receiveHeaderSTR(clientSocket);
            // Variables that will contain the header data after parsing
            std::string receivedMD5, machineID;
            int64_t fileSize;
            // Receive and parse header from client
            parseHeader(headerString, receivedMD5, machineID, fileSize);
            // Remove newline from the string
            std::string sanitizedID = sanitizeMID(machineID);
            // Debugging
            std::cout << count << ": Header Received: MD5: " << receivedMD5 << " ID: " << sanitizedID << " File Size: " << fileSize << std::endl;

            // Send ack message
            send(clientSocket, ackMessage, strlen(ackMessage), 0);

            // Receive data from client
            // Create temp char* buffer to receive bytes
            char* clData = new char[fileSize];
            // Receive data to temp buffer
            receiveBufferSock(clientSocket, clData, fileSize);


            // Store data from temp buffer to appropriate client buffer
            SharedBuffer* cBuffer = (*clientBuffers).getBuffer(stringToInt(sanitizedID));
            //std::cout << sanitizedID << std::endl;
            if (cBuffer==NULL){
                perror("Invalid machine ID, terminating...");
                exit(1);
            }
            cBuffer->write(clData, fileSize);
            // Notify server that writing is done
            cBuffer->setComplete();
            // Free temp buffer
            delete[] clData;
            std::cout << count << ": File received from machine with ID: " << sanitizedID << std::endl;
            // Send ack message,
            send(clientSocket, ackMessage, strlen(ackMessage), 0);

            // ------------ Wait until the aggregated data is ready ------------
            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this]{ return aggregation.load(); });
            lock.unlock();

            // Read aggregation buffer set by Python
            // Get size and md5 of aggregated data
            SharedBuffer* aBuffer = (*aggregationBuffer).getBuffer(stringToInt(sanitizedID));
            int buffSize = aBuffer->size();
            std::string aggregatedMD5 = aBuffer->md5_str;

            // Send header of aggregated file to client
            std::string header_aggregated = setupHeader(aggregatedMD5, stringToInt(sanitizedID), buffSize);
            send(clientSocket, header_aggregated.c_str(), header_aggregated.size(), 0);
            //std::cout << count << ": Header Sent: " << header_aggregated;
            std::cout << count << ": Header Sent: MD5: " << aggregatedMD5 << " ID: " << sanitizedID << " File Size: " << buffSize << std::endl;

            //Receive ack
            memset(ackBuffer, 0, sizeof(ackBuffer));
            recv(clientSocket, ackBuffer, sizeof(ackBuffer), 0);

            // Send data to client
            char* aggData = new char[buffSize];
            aBuffer->read(aggData, buffSize, 0);

            sendBufferSock(clientSocket, aggData, buffSize);
            delete[] aggData;

            //Receive ack, terminate
            memset(ackBuffer, 0, sizeof(ackBuffer));
            recv(clientSocket, ackBuffer, sizeof(ackBuffer), 0);
            std::cout << count << ": Sent aggregated file to machine with ID: " << sanitizedID << std::endl;


            // Give lock to next thread
//            lock.unlock();
            // Increment rounds count
            count++;

            // Wait until the the aggregation flag is reset
            (*globalBarrier).wait();
            std::cout << "Ending loop, machine with ID: " << sanitizedID << std::endl;

            // Decide whether to stop the rounds based on the server message inside aggregated data
            if(stop_rounds.load() == true) break;
        }
        //Notify that all work is done for this thread
        completedJobs.fetch_add(1, std::memory_order_relaxed);
#ifdef _WIN32
        closesocket(clientSocket);
#else
        close(clientSocket);
#endif
    }

public:
    // To be called by Python, in order to wait for all the detached threads to complete.
    void allDone() {
        std::mutex done_mtx;
        std::unique_lock<std::mutex> lock_t(done_mtx);
        while(1) {
            if (completedJobs >= (numOfClients+1)) {
                // When all clients have finished, exit
                std::cout << "All jobs completed, terminating..." << std::endl;
                return;
            }
            // Sleep, in order to not waste cycles on checks
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    // Reset aggregation condition
    void aggregationReset(void) {
        aggregation.store(false, std::memory_order_release);
        std::cout << "Aggregation data condition reset." << std::endl;
    }

    // Notify all threads to wake them up, called externally by Python
    void aggregationDone(bool s_rounds) {
        aggregation.store(true, std::memory_order_release);
        if (s_rounds) {
            stop_rounds.store(true, std::memory_order_release);
        } else {
            stop_rounds.store(false, std::memory_order_release);
        }
        cv.notify_all();
    }



    // Initialize Barrier object that resets aggregation condition when all threads enter it
    void initializeGlobalBarrier() {
        auto completionFunction = std::bind(&FL_Server::aggregationReset, this);
        globalBarrier.reset(new Barrier((numOfClients+1), completionFunction)); // Initialize with actual barrier object
    }

    // Run the server until clientCount > numOfClients
    void run() {
        int clientSocket;
        // Setup a Barrier in order to reset the aggregation data
        initializeGlobalBarrier(); // Make sure this is called before using the globalBarrier
        // Bind the member function to the instance `task` and create a std::function

        std::cout << "Master server is running on port " << masterPort << std::endl;
        while (clientCount <= numOfClients) {
            // Accept unconditionally
            clientSocket = accept(server_fd, NULL, NULL);
            if (clientSocket < 0) {
                perror("accept failed");
                continue;
            }
            // Increment clientCount for new client connection
            clientCount.fetch_add(1, std::memory_order_relaxed);
            // Client 1 starts with index 0 and so on
            std::thread clientThread(&FL_Server::handleClientConnection, this, clientSocket, (clientCount-1));
            clientThread.detach();
        }
    }

    // FL_Server constructor
    FL_Server(int port,int numOfClients, BufferManager* agg_bm, BufferManager* clientBufferManager) :
    masterPort(port), numOfClients(numOfClients), clientCount(0), completedJobs(0),
    aggregationBuffer(agg_bm), clientBuffers(clientBufferManager)
    {
        aggregation.store(false, std::memory_order_release);
        stop_rounds.store(false, std::memory_order_release);

#ifdef _WIN32
        // Initialize Winsock for Windows
        WSADATA wsaData;
        int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (result != 0) {
            printError(-1, "WSAStartup failed");
        }
#endif

        // Initialize Buffer Manager objects with numOfClients+1 objects
        (*clientBuffers).createBuffers(numOfClients+1);
        (*aggregationBuffer).createBuffers(numOfClients+1);
        // Creating socket file descriptor
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd == -1) {
           printError(server_fd, "Socket creation error");
        }

        // Forcefully attaching socket to the port
#ifdef _WIN32
        const char opt_val = 1;
        if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val)) < 0) {
#else
        if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt)) < 0) {
#endif
            printError(server_fd, "setsockopt error");
        }

        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(masterPort);

        // Forcefully attaching socket to the port
        if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
            printError(server_fd, "Bind failed - Server Listener Socket");
        }
        if (listen(server_fd, SOMAXCONN) < 0) {
            printError(server_fd, "Listen failed");
        }
        // Initialize barrier
        globalBarrier = nullptr;
    }

    // Destructor - for garbage collection
    ~FL_Server() {
#ifdef _WIN32
        closesocket(new_socket);
        closesocket(server_fd);
        WSACleanup();
#else
        close(new_socket);
        close(server_fd);
#endif
    }
};