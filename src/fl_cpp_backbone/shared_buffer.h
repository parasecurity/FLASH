#pragma once

#include <vector>
#include <mutex>
#include <string>

class SharedBuffer {

private:
    // Use mutex for thread-safe access
    mutable std::mutex mtx;
    // Use std::vector for security and dynamic allocation
    std::vector<char> buffer;

public:
    // Use this variable to set process complete status
    bool writing_complete;
    // Use this variable to store MD5 checksum of buffer contents after writing is complete
    std::string md5_str;

    void write(const char* data, size_t length);
    void read(char* output, size_t length, size_t offset);
    size_t size(void) const;
    char* getBufferPtr(void);
    void clear(void);
    bool checkComplete(void);
    void setComplete(void);
    void resetComplete(void);
    void setMD5(const char* data);
};

class BufferManager {

private:
    // Keep a vector of Client buffers
    std::vector<SharedBuffer*> buffers;
    // Use mutex for thread-safe access
    mutable std::mutex mtx;
public:
    ~BufferManager();
    // Method to create and add Buffer objects to the vector
    void createBuffers(int count);
    // Access a Buffer by index
    SharedBuffer* getBuffer(int index);
    // Get the number of Buffer objects
    size_t size() const;

};
