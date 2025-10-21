#include "shared_buffer.h"
#include <iostream>
#include <cstring>

// ---------------------- Shared Buffer methods  ---------------------- //

void SharedBuffer::write(const char* data, size_t length) {
//    std::lock_guard<std::mutex> guard(mtx);
    buffer.insert(buffer.end(), data, data + length);
}

void SharedBuffer::read(char* output, size_t length, size_t offset) {
//    std::lock_guard<std::mutex> guard(mtx);
    if (offset + length > buffer.size()) {
        std::cerr << "Read operation exceeds buffer bounds." << std::endl;
        return;
    }
    std::memcpy(output, buffer.data() + offset, length);
}

// Get buffer capacity
size_t SharedBuffer::size(void) const {
//    std::lock_guard<std::mutex> guard(mtx);
    return buffer.size();
}

// Method to get a pointer to the buffer
char* SharedBuffer::getBufferPtr(void) {
//    std::lock_guard<std::mutex> guard(mtx);
    return buffer.data();
}

// Clear buffer data
void SharedBuffer::clear() {
//    std::lock_guard<std::mutex> guard(mtx);
    buffer.clear();
    SharedBuffer::writing_complete = false;
}

// Check if writing is complete
bool SharedBuffer::checkComplete(void) {
//    std::lock_guard<std::mutex> guard(mtx);
    return SharedBuffer::writing_complete;
}

// Set writing as complete
void SharedBuffer::setComplete(void) {
//    std::lock_guard<std::mutex> guard(mtx);
    SharedBuffer::writing_complete = true;
}

// Set writing as not complete
void SharedBuffer::resetComplete(void) {
//    std::lock_guard<std::mutex> guard(mtx);
    SharedBuffer::writing_complete = false;
}

// Set md5 checksum of buffer to given value
void SharedBuffer::setMD5(const char* data) {
//    std::lock_guard<std::mutex> guard(mtx);
    SharedBuffer::md5_str = data;
}

// ---------------------- Buffer Manager methods  ---------------------- //

// Clean up allocated SharedBuffers
BufferManager::~BufferManager() {
    for (auto buffer : buffers) {
        delete buffer;
    }
}

void BufferManager::createBuffers(int count) {
    // Dynamically allocate SharedBuffer objects once
    for (int i = 0; i < count; ++i) {
        buffers.push_back(new SharedBuffer());
    }
}

SharedBuffer* BufferManager::getBuffer(int index) {
    std::lock_guard<std::mutex> guard(mtx);
    if (index < 0 || static_cast<size_t>(index) >= buffers.size()) {
        // Handle out-of-range index
        return nullptr;
    }
    return buffers[index]; // Directly return the pointer from the vector
}

size_t BufferManager::size(void) const {
    std::lock_guard<std::mutex> guard(mtx);
    return buffers.size();
}
