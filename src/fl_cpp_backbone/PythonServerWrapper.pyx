from libcpp.string cimport string
from libc.stdint cimport uintptr_t
# from libcpp.mutex cimport mutex
#from libcpp.condition_variable cimport condition_variable
from libc.stdlib cimport malloc, free

cdef extern from "server.cpp":
    cdef cppclass FL_Server:
        FL_Server(int port, int numOfClients, BufferManager* agg_bm, BufferManager* bm) except +
        void run()
        void aggregationDone(bint s_rounds)
        void allDone()
        SharedBuffer* getCorrespondingBuffer(int mid)

    # Mutex locks will happen inside the c++ implementation
    cdef cppclass SharedBuffer:
        SharedBuffer() except +
        void write(const char*, size_t)
        void read(char*, size_t, size_t)
        size_t size()
        char* getBufferPtr()
        void clear()
        bint checkComplete()
        void resetComplete()
        void setMD5(const char*)
        bint writing_complete
        string md5_str


    cdef cppclass BufferManager:
        BufferManager()  # Assuming default constructor exists
        void createBuffers(int count)
        SharedBuffer* getBuffer(int index)
        size_t size()


cdef class py_fl_server:
    # pointer to TCPServer object
    cdef FL_Server* _server
    # cdef SharedBuffer* c_sharedBuffer

    def __cinit__(self, int port, int numOfClients, PyBufferManager buffer, PyBufferManager bm):
        # Give the pointer of buffer as initializer
        self._server = new FL_Server(port,numOfClients,buffer.c_manager, bm.c_manager)

    def __dealloc__(self):
        del self._server

    def run(self):
        self._server.run()

    def aggregationDone(self, s_rounds):
        self._server.aggregationDone(s_rounds)

    def allDone(self):
        self._server.allDone()

cdef class PyServerBuffer:
    cdef SharedBuffer * c_sharedBuffer

    def __cinit__(self):
        self.c_sharedBuffer = new SharedBuffer()
        self.c_sharedBuffer.writing_complete = False

    def __dealloc__(self):
        del self.c_sharedBuffer

    def write(self, bytes data):
        self.c_sharedBuffer.write(data, len(data))

    def read(self, int length, int offset=0):
        cdef char * output = <char *> malloc(length)
        try:
            self.c_sharedBuffer.read(output, length, offset)
            return bytes(output[:length])
        finally:
            free(output)

    def size(self):
        return self.c_sharedBuffer.size()

    def getBufferPtr(self):
        return <uintptr_t> self.c_sharedBuffer.getBufferPtr()

    def set_md5(self, bytes data):
        self.c_sharedBuffer.setMD5(data)
    def check_complete(self):
        return self.c_sharedBuffer.checkComplete()
    def clear(self):
        self.c_sharedBuffer.clear()
        # self.c_sharedBuffer.resetComplete()

cdef class PyBufferManager:
    cdef BufferManager* c_manager
    def __cinit__(self):
        self.c_manager = new BufferManager()

    def __dealloc__(self):
        del self.c_manager

    def create_buffers(self, int count):
        self.c_manager.createBuffers(count)

    # Will only return the data of the buffer in python
    def get_buffer(self, int index, int length):
        cdef SharedBuffer * sb = self.c_manager.getBuffer(index)
        cdef char * buffer = <char *> malloc(length * sizeof(char))
        # if not buffer:
        #     raise MemoryError("Failed to allocate read buffer")
        try:
            # Read data into the buffer
            sb.read(buffer, length, 0)  # Assuming offset is 0

            # Convert to Python bytes
            return bytes(buffer[:length])
        finally:
             free(buffer)
    def set_buffer(self, int index, bytes data):
        cdef SharedBuffer * sb = self.c_manager.getBuffer(index)
        sb.write(data, len(data))

    def set_md5(self, int index, bytes data):
        cdef SharedBuffer * sb = self.c_manager.getBuffer(index)
        sb.setMD5(data)

    def get_buffer_size(self, int index):
        return self.c_manager.getBuffer(index).size()

    def check_buffer_complete(self, int index):
        return self.c_manager.getBuffer(index).checkComplete()

    def clear_all_buffers(self):
        for i in range(self.c_manager.size()):
            # print(f"Clearing buffer {i}")
            self.c_manager.getBuffer(i).clear()

    def __len__(self):
        return self.c_manager.size()