from FL_cpp_server import PyServerBuffer, PyBufferManager
from FL_cpp_client import PyClientBuffer
from utils.helper import *

class BufferIO:
    def __init__(self, role: str):
        self.role = role
        self.symmetric_key = None
        self.max_id = None
        # Initialize appropriate buffers
        if role == "server":
            # outputBuffer contains the aggregation data to be sent to all clients
            self.outputBuffer = PyBufferManager()
            # clientDataBuffers is a manager object that handles one buffer for each client
            self.inputBuffers = PyBufferManager()
        elif role == "client":
            # Client input buffer
            self.aggregationBuffer = PyClientBuffer()
            # Client output buffer
            self.paramsBuffer = PyClientBuffer()
        else:
            raise ValueError("Invalid role, must be either server or client.")

    # Set symmetric keys array
    def setKeys(self, symmetric_key_array, max_mid):
        if self.role == 'server':
            self.symmetric_key = symmetric_key_array
            self.max_id = max_mid
        else:
            raise ValueError("THiss function call is only allowed for server")


    def send(self, data, max_id=None, rsa_key=None):
        # Handle client to server send
        if self.role == 'client':
            pBytes: bytes = dill.dumps(data)
            tempByteIO: BytesIO = BytesIO(pBytes)
            print(f"Size of params in bytes: {tempByteIO.getbuffer().nbytes}")
            pBuffMD5 = get_md5_checksum_from_bytesio(tempByteIO)
            print("MD5: " + pBuffMD5)
            tempByteIO.seek(0)
            # Send encrypted data, after sending the RSA public key
            if self.symmetric_key is not None:
                eBytes = encrypt_and_prepare(pBytes, self.symmetric_key)
                pBytes = eBytes
            self.paramsBuffer.write(pBytes)
            self.paramsBuffer.set_md5(pBuffMD5.encode())
            self.paramsBuffer.set_complete()
            # Clear aggregation buffer
            self.aggregationBuffer.clear()
        # TODO Handle server to client send
        elif self.role == 'server':
            # Reset the aggregation data buffer before writing
            self.outputBuffer.clear_all_buffers()
            # Send RSA-encrypted symmetric keys
            if rsa_key is not None:
                for mid in range(max_id):
                    eBytes = encrypt_message_rsa(self.symmetric_key[mid], rsa_key[mid])
                    self.outputBuffer.set_buffer(mid, eBytes)
                # self.inputBuffers.clear_all_buffers()
                return
            # Send data with or without encryption
            bytes = dill.dumps(data)
            tempByteIO = BytesIO(bytes)
            print("MD5: " + get_md5_checksum_from_bytesio(tempByteIO))
            tempByteIO.seek(0)
            # In the first run self.max_id will be set
            if max_id is None: max_id = self.max_id
            # Write to all output buffers
            for mid in range(max_id):
                #  If the key is given, the data should be encrypted
                if self.symmetric_key is not None:
                    eBytes = encrypt_and_prepare(bytes, self.symmetric_key[mid])
                    self.outputBuffer.set_buffer(mid, eBytes)
                else:
                    self.outputBuffer.set_buffer(mid, bytes)
                self.outputBuffer.set_md5(mid, get_md5_checksum_from_bytesio(tempByteIO).encode())
            print(f"Size of parameters in bytes: {tempByteIO.getbuffer().nbytes}")
            # self.inputBuffers.clear_all_buffers()
            return

    def receive(self, max_id=None, rsa_key=None):
        # Receive data from server
        if self.role == 'client':
            while not self.aggregationBuffer.check_complete():
                # print("waiting for new params...")
                time.sleep(0.1)
            # Create BytesIO object
            clientBufferIO = BytesIO()
            # Get buffer size for read and write read data to BytesIO object
            buffSize = self.aggregationBuffer.size()
            clientBufferIO.write(self.aggregationBuffer.read(buffSize))
            aBuffMD5 = get_md5_checksum_from_bytesio(clientBufferIO)
            print("MD5: " + aBuffMD5)
            # Reset the BytesIO pointer
            clientBufferIO.seek(0)
            # If no RSA key is given and no symmetric key is registered, data is not encrypted
            if self.symmetric_key is None and rsa_key is None:
                # Load the serialized data to memory
                params = dill.load(clientBufferIO)
            # Receive symmetric key from server by using RSA decryption
            elif self.symmetric_key is None and rsa_key is not None:
                encrypted_symmetric_key = clientBufferIO.getvalue()
                decrypted_symmetric_key = decrypt_message_rsa(encrypted_symmetric_key, rsa_key)
                # Keep symmetric key in ClientIO
                self.symmetric_key = decrypted_symmetric_key
                self.aggregationBuffer.clear()
                return decrypted_symmetric_key
            # Decrypt data (after receiving symmetric key communication is encrypted)
            elif self.symmetric_key is not None and rsa_key is None:
                eBytes = clientBufferIO.getvalue()
                dBytes = receive_and_decrypt(eBytes, self.symmetric_key)
                dBytesIO = BytesIO(dBytes)
                # Load the serialized data to memory
                params = dill.load(dBytesIO)
            else:
                raise ValueError("Invalid key configuration.")
            # Clean the buffers, everything is read
            self.aggregationBuffer.clear()
            # In cases where params dictionary is filled, return params
            return params

        # Receive data from clients
        elif self.role == 'server':
            # self.inputBuffers.clear_all_buffers()
            params = []
            if max_id is None: max_id = self.max_id
            for mid in range(max_id):
                # Check if buffer data writing is complete, else wait for it to finish
                while not self.inputBuffers.check_buffer_complete(mid):
                    time.sleep(0.1)
                clientBufferIO = BytesIO()
                buffSize = self.inputBuffers.get_buffer_size(mid)
                clientBufferIO.write(self.inputBuffers.get_buffer(mid, buffSize))
                # Reset the BytesIO pointer
                clientBufferIO.seek(0)
                # If symmetric key array is set then decrypt data
                if self.symmetric_key is not None:
                    # Decrypt data using symmetric keys
                    eBytes = clientBufferIO.getvalue()
                    dBytes = receive_and_decrypt(eBytes, self.symmetric_key[mid])
                    dBytesIO = BytesIO(dBytes)
                    params.append(dill.load(dBytesIO))
                # If symmetric key array is None then data is not encrypted
                else:
                    params.append(dill.load(clientBufferIO))
            self.inputBuffers.clear_all_buffers()
            return params


