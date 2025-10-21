// Barrier.cpp
#include "barrier.h"
#include <iostream> // For std::cout

Barrier::Barrier(std::size_t count, std::function<void()> completionFunction)
    : thread_count(count), count(count), generation(0), completionFunction(completionFunction) {}

void Barrier::wait() {
    std::unique_lock<std::mutex> lock(mutex);
    // Capture the current generation on local variable gen
    std::size_t gen = generation;
    if (--count == 0) {
        // Call the completion function just before allowing threads to proceed
        completionFunction();
        // Reset the barrier to its initial state
        generation++;
        count = thread_count;
        // Notify all the threads to reset
        cond.notify_all();
    } else {
        // Only proceed on the correct generation
        cond.wait(lock, [this, gen] { return gen != generation; });
    }
}
