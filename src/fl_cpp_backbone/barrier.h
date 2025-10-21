#pragma once
// Imports
#include <functional>
#include <mutex>
#include <condition_variable>

// Emulates c++20 barrier, has wait function that waits for a certain num of threads to arrive
class Barrier {
public:
    explicit Barrier(std::size_t count, std::function<void()> completionFunction = [](){});
    void wait();

private:
    std::mutex mutex;
    std::condition_variable cond;
    std::size_t thread_count;
    std::size_t count;
    std::size_t generation;
    std::function<void()> completionFunction;
};
