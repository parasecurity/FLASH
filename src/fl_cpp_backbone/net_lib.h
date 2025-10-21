#ifndef NETLIB_H
#define NETLIB_H

// Platform-specific network includes
#ifdef _WIN32
    // Windows headers
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <windows.h>
    #pragma comment(lib, "ws2_32.lib")
    
    // Define Unix-style functions for Windows
    typedef int socklen_t;
    // Don't define close as closesocket - it conflicts with other Windows functions
    
    // Windows doesn't have unistd.h, but we need some of its functionality
    #include <io.h>
    #include <process.h>
#else
    // Unix/Linux headers
    #include <sys/socket.h>
    #include <arpa/inet.h>
    #include <netinet/in.h>
    #include <unistd.h>
#endif

// Handle all the required includes in this lib files
// Network - already included above in platform-specific section

// General
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <set>
#include <mutex>
#include <filesystem>

// Multithreading
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>

// String manipulation
#include <iomanip> // For std::setw and std::setfill

// Additional includes for cross-platform compatibility
#include <algorithm> // For std::min
#include <climits>   // For INT_MAX

// Define some constants
extern const char* ackMessage;
// Declare some helper methods


int stringToInt(const std::string& str);
// setup header structure
std::string setupHeader(const std::string md5_checksum, int machine_id, int64_t file_size);
// Parse the header string into member variables
void parseHeader(const std::string& header, std::string& receivedMD5, std::string& machineID, int64_t& fileSize);
// Method to simplify printing error message and exiting after socket operation
void printError(int sock, const std::string& errorMessage);
// remove newline character from the machineID string
std::string sanitizeMID(const std::string& input);
// Get file size by seeking to the end
int getFileSize(const std::string filename);
// Receive file from socket and write it
void receiveFileSock(int socket, std::string filename, int64_t fileSize);
// Read file and send to socket
void sendFileSock(int socket, std::string filename, int64_t fileSize);
// Function to send data to socket from a buffer
void sendBufferSock(int socket, char* dataBuffer, int64_t bufferSize);
// Function to receive data from socket and store it in a provided buffer
void receiveBufferSock(int socket, char* dataBuffer, int64_t bufferSize);
// Receive header string from socket
std::string receiveHeaderSTR(int sock);

#endif