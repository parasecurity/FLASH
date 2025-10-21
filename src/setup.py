# python3 setup.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize
import platform

# Platform-specific settings
extra_compile_args = ["-std=c++11"]
extra_link_args = []
libraries = []

if platform.system() == "Windows":
    # Windows specific settings
    extra_compile_args = ["/std:c++17"]  # Need C++17 for filesystem
    libraries = ["ws2_32"]  # Windows Socket library
    extra_link_args = []
else:
    # Unix-like systems (Linux, macOS)
    extra_compile_args = ["-std=c++17"]  # Also update to C++17 for consistency
    extra_link_args = []
    libraries = []

extensions = [
    Extension("FL_cpp_server",
              ["fl_cpp_backbone/PythonServerWrapper.pyx", "fl_cpp_backbone/server.cpp", "fl_cpp_backbone/net_lib.cpp",
               "fl_cpp_backbone/shared_buffer.cpp", "fl_cpp_backbone/barrier.cpp"],
              language='c++',
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              libraries=libraries),

    Extension("FL_cpp_client",
              ["fl_cpp_backbone/PythonClientWrapper.pyx", "fl_cpp_backbone/client.cpp", "fl_cpp_backbone/net_lib.cpp",
               "fl_cpp_backbone/shared_buffer.cpp"],
              language='c++',
              extra_compile_args=extra_compile_args,
              extra_link_args=extra_link_args,
              libraries=libraries),
]

setup(
    name="FLASH_framework",
    ext_modules=cythonize(extensions, language_level="3"),
    license="Apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
    ],
)

print("Successfully built.")