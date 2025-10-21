// Define some common helper methods and constants for the server and client
#include "net_lib.h"

const char* ackMessage = "ACK";

int stringToInt(const std::string& str) {
    std::istringstream iss(str);
    int result;
    if (!(iss >> result)) {
        // Handle conversion failure
        throw std::invalid_argument("stoi alternative: Invalid argument");
    }
    return result;
}

// setup header structure
std::string setupHeader(const std::string md5_checksum, int machine_id, int64_t file_size) {
    std::ostringstream stream;
    // MD5 checksum part remains unchanged
    stream << "MD5:" << md5_checksum;
    // Pad the machine_id with leading zeros to ensure it is 5 characters long
    stream << ";ID:" << std::setw(5) << std::setfill('0') << machine_id;
    // Pad the file_size with leading zeros to ensure it is 10 characters long
    stream << ";SIZE:" << std::setw(10) << std::setfill('0') << file_size;
    // Add the newline character at the end
    stream << "\n";
    return stream.str();
}

// Parse the header string into member variables
void parseHeader(const std::string& header, std::string& receivedMD5, std::string& machineID, int64_t& fileSize) {
    std::istringstream headerStream(header);
    std::string segment;
    // Split the header string by ';'
    while (std::getline(headerStream, segment, ';')) {
        std::size_t delimiterPos = segment.find(':');
        if (delimiterPos != std::string::npos) {
            std::string key = segment.substr(0, delimiterPos);
            std::string value = segment.substr(delimiterPos + 1);
            if (key == "MD5") {
                receivedMD5 = value;
            } else if (key == "ID") {
                machineID = value;
            } else if (key == "SIZE") {
                fileSize = std::stoll(value);
            }
        }
    }
}

// Method to simplify printing error message and exiting after socket operation
void printError(int sock, const std::string& errorMessage) {
    std::cout << "\n" << errorMessage << "\n";
#ifdef _WIN32
    if (sock != -1) {
        closesocket(sock);
    }
    // Print Windows socket error if applicable
    int error = WSAGetLastError();
    if (error != 0) {
        std::cerr << "WSA Error Code: " << error << std::endl;
    }
#else
    if (sock != -1) {
        close(sock);
    }
#endif
    exit(EXIT_FAILURE);
}

// remove newline character from the machineID string
std::string sanitizeMID(const std::string& input) {
    std::string output;
    for (char c : input) if (c != '\n') output += c;
    return output;
}

// Get file size by seeking to the end
int getFileSize(const std::string filename) {
    // Open file in binary mode and set the position to the end
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    // We seek to the end so the position is the size
    std::streamsize fSize = file.tellg();
    file.close();
    return static_cast<int>(fSize);  // Explicit cast to avoid warning
}

// Receive data from socket and write it to disk
void receiveFileSock(int socket, std::string filename, int64_t fileSize) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile.is_open()) {
#ifdef _WIN32
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
#else
        perror("Failed to open file for writing");
#endif
        exit(EXIT_FAILURE);
    }
    char buffer[1024];
    int64_t total_bytes_read = 0;
    
    while (total_bytes_read < fileSize) {
#ifdef _WIN32
        int bytes_read = recv(socket, buffer, 
                             static_cast<int>(min(static_cast<int64_t>(sizeof(buffer)), fileSize - total_bytes_read)), 0);
#else
        int bytes_read = read(socket, buffer, 
                             min(sizeof(buffer), static_cast<size_t>(fileSize - total_bytes_read)));
#endif
        if (bytes_read <= 0) {
            break;
        }
        outfile.write(buffer, bytes_read);
        total_bytes_read += bytes_read;
    }
    outfile.close();
}

// Read file from disk and send it to socket
void sendFileSock(int socket, std::string filename, int64_t fileSize) {
    int64_t total_bytes_sent = 0;
    char buffer[1024];
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) {
#ifdef _WIN32
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
#else
        perror("Failed to open file for reading");
#endif
        return;
    }
    
    while (total_bytes_sent < fileSize) {
        infile.read(buffer, sizeof(buffer));
        int bytes_read = static_cast<int>(infile.gcount());  // Number of bytes read from file
        
        // Check if we have read any bytes, and if so, send them
        if (bytes_read > 0) {
#ifdef _WIN32
            int bytes_sent = send(socket, buffer, bytes_read, 0);
#else
            int bytes_sent = send(socket, buffer, bytes_read, 0);
#endif
            if (bytes_sent <= 0) {
                break; // Handle errors
            }
            total_bytes_sent += bytes_sent;
        }
    }
    infile.close();
}

// Function to send data to socket from a buffer
void sendBufferSock(int socket, char* dataBuffer, int64_t bufferSize) {
    int64_t total_bytes_sent = 0;

    while (total_bytes_sent < bufferSize) {
#ifdef _WIN32
        // Windows send() takes int for length
        int bytes_to_send = static_cast<int>(min(static_cast<int64_t>(INT_MAX), bufferSize - total_bytes_sent));
        int bytes_sent = send(socket, dataBuffer + total_bytes_sent, bytes_to_send, 0);
#else
        ssize_t bytes_sent = send(socket, dataBuffer + total_bytes_sent, bufferSize - total_bytes_sent, 0);
#endif
        if (bytes_sent <= 0) {
            break; // Handle errors
        }
        total_bytes_sent += bytes_sent;
    }
    //std::cout << "Sent: "<< std::to_string(total_bytes_sent) << " bytes." << std::endl;
}

// Function to receive data from socket and store it in a provided buffer
void receiveBufferSock(int socket, char* dataBuffer, int64_t bufferSize) {
    int64_t total_bytes_read = 0;
    
    while (total_bytes_read < bufferSize) {
#ifdef _WIN32
        // Windows recv() takes int for length
        int bytes_to_read = static_cast<int>(min(static_cast<int64_t>(INT_MAX), bufferSize - total_bytes_read));
        int bytes_read = recv(socket, dataBuffer + total_bytes_read, bytes_to_read, 0);
#else
        ssize_t bytes_read = read(socket, dataBuffer + total_bytes_read, bufferSize - total_bytes_read);
#endif
        if (bytes_read <= 0) {
            // Handle errors
            break;
        }
        total_bytes_read += bytes_read;
    }
    //std::cout << "Received: "<< std::to_string(total_bytes_read) << " bytes." << std::endl;
}

// Receive header string from socket
std::string receiveHeaderSTR(int sock) {
    char headerBuffer[62] = {0};
#ifdef _WIN32
    int headerLength = recv(sock, headerBuffer, sizeof(headerBuffer), 0);
#else
    int headerLength = recv(sock, headerBuffer, sizeof(headerBuffer), 0);
#endif
    // std::cout << "Received header of size: "<< std::to_string(headerLength) << " bytes." << std::endl;
    std::string header(headerBuffer, headerLength);
    return header;
}