from libcpp.string cimport string
from libc.stdint cimport int64_t
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free

cdef extern from "client.cpp":
    cdef cppclass FL_Client:
        FL_Client(const char* server_ip, int server_port, SharedBuffer* paramsBuffer, SharedBuffer* aggregationBuffer) except +
        void participate()
        void parseHeaderWrapper(const string& header, string& receivedMD5, string& machineID, int64_t& fileSize)
        void setParamsReady(bint s_rounds)
        int getMachineID()

    cdef cppclass SharedBuffer:
        SharedBuffer() except +
        void write(const char *, size_t)
        void read(char *, size_t, size_t)
        size_t size()
        char * getBufferPtr()
        void clear()
        bint checkComplete()
        void setComplete()
        void setMD5(const char *)
        bint writing_complete
        string md5_str

cdef class py_fl_client:
    cdef FL_Client* _client

    def __cinit__(self, server_ip, int server_port, PyClientBuffer paramsBuffer, PyClientBuffer aggregationBuffer):
        self._client = new FL_Client(server_ip.encode(), server_port, paramsBuffer.c_sharedBuffer, aggregationBuffer.c_sharedBuffer)

    def __dealloc__(self):
        del self._client

    def participate(self):
        self._client.participate()
    def getMachineID(self):
        return self._client.getMachineID()

    def parse_header(self,header):
        # Init string vars and int var
        cdef string c_receivedMD5
        cdef string c_machineID
        cdef int64_t c_fileSize

        # Call the C++ function
        self._client.parseHeaderWrapper(header, c_receivedMD5, c_machineID, c_fileSize)

        # Convert C++ strings back to Python strings and return a Python tuple
        return c_receivedMD5.decode('utf-8'), c_machineID.decode('utf-8'), c_fileSize

    def setParamsReady(self, s_rounds):
        self._client.setParamsReady(s_rounds)

    # def setParamsReady(self):
    #     self._client.setParamsReady()

cdef class PyClientBuffer:
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
    def set_complete(self):
        return self.c_sharedBuffer.setComplete()
    def clear(self):
        self.c_sharedBuffer.clear()
        # self.c_sharedBuffer.resetComplete()