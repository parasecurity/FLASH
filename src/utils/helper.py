import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import os, sys, dill, time, socket
import numpy as np
import pandas as pd
import threading
import json
from io import BytesIO
from os import urandom
import base64

# Python cryptography library
# https://cryptography.io/en/latest/hazmat/primitives/asymmetric/rsa/
# https://cryptography.io/en/latest/hazmat/primitives/symmetric-encryption/
# https://crypto.stackexchange.com/questions/62228/generate-shared-secrets-using-rsa

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import padding as symmetric_padding


### Helper Methods ###
def get_md5_checksum(file_path):
    """ Hashlib default md5 checksum method """
    md5 = hashlib.md5()
    with open(file_path, "rb") as file:
        md5.update(file.read())
    return md5.hexdigest()


def get_md5_checksum_from_bytesio(bytes_io):
    """Calculate MD5 checksum for data in a BytesIO object"""
    md5 = hashlib.md5()
    # Ensure the BytesIO object's pointer is at the start
    bytes_io.seek(0)
    # Update the MD5 hash with the content of the BytesIO object
    md5.update(bytes_io.read())
    # Return the hexadecimal MD5 digest
    return md5.hexdigest()


def parse_header(header):
    """ Get info from received header tuple """
    # Assuming header is a bytes object, decode it to a string
    header_str = header.decode('utf-8')
    # Split the header string by ';'
    segments = header_str.split(';')
    for segment in segments:
        # Split each segment into key and value
        if ':' in segment:
            key, value = segment.split(':', 1)
            if key == 'MD5':
                md5 = value.strip()
            elif key == 'ID':
                machine_id = value.strip()
    return md5, machine_id



def generate_rsa_keys():
    private_key = rsa.generate_private_key(
        # Almost everyone should use 65537.
        public_exponent=65537,
        # It is strongly recommended to be at least 2048
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    # Serialize keys to make them transportable
    pem_private_key = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    pem_public_key = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    return pem_private_key, pem_public_key


# Generate 256-bit symmetric key
def generate_aes_key():
    return urandom(32)


# Asymmetric RSA encryption of a message using the public key
def encrypt_message_rsa(message, pem_public_key):
    public_key = serialization.load_pem_public_key(
        pem_public_key,
        backend=default_backend()
    )

    return public_key.encrypt(
        message,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )


# Asymmetric RSA decryption of a message using the private key
def decrypt_message_rsa(encrypted_message, pem_private_key):
    private_key = serialization.load_pem_private_key(
        pem_private_key,
        password=None,
        backend=default_backend()
    )

    return private_key.decrypt(
        encrypted_message,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        return super().default(obj)

def save_var_to_json(var, name, dir):
    with open(os.path.join(dir, name), 'w') as f:
        json.dump(var, f, cls=NumpyEncoder)

# Symmetric AES encryption and base64 encode method that creates a payload that also contains the generated iv
def encrypt_and_prepare(data, key):
    # Generate a cryptographically secure IV
    iv = os.urandom(16)  # 16 bytes for AES block size

    # Create an AES CBC cipher instance with the given key and IV
    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend()).encryptor()

    # Pad the data using PKCS7 padding to ensure it's a multiple of the block size
    padder = symmetric_padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()

    # Encrypt the data
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    # Concatenate the IV with the encrypted data and encode it to base64 for easy transmission
    payload = base64.b64encode(iv + encrypted_data)

    return payload


# Symmetric AES decryption and base64 decode method that expects a payload that also contains the generated iv
def receive_and_decrypt(payload, key):
    combined = base64.b64decode(payload)
    iv = combined[:16]              # The first 16 bytes are the IV
    encrypted_data = combined[16:]  # The rest is the encrypted data
    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend()).decryptor()
    decrypted_padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
    unpadder = symmetric_padding.PKCS7(algorithms.AES.block_size).unpadder()
    decrypted_data = unpadder.update(decrypted_padded_data) + unpadder.finalize()
    return decrypted_data
