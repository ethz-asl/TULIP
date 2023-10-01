
from setuptools import setup, find_packages

setup(
    name="voxelizer",
    version="0.1",
    packages=find_packages(),
)

# import os
# import re
# import sys
# import platform
# import subprocess
# from setuptools import setup, Extension, find_packages
# from setuptools.command.build_ext import build_ext

# class CMakeExtension(Extension):
#     def __init__(self, name, sourcedir=''):
#         Extension.__init__(self, name, sources=[])
#         self.sourcedir = os.path.abspath(sourcedir)

# class CMakeBuild(build_ext):
#     def run(self):
#         for ext in self.extensions:
#             self.build_cmake(ext)

#     def build_cmake(self, ext):
#         cwd = os.getcwd()

#         # these dirs will be created in build
#         build_temp = os.path.abspath(os.path.join(self.build_temp, ext.name))
#         build_lib = os.path.abspath(self.build_lib)

#         # create temp and lib directories
#         if not os.path.exists(build_temp):
#             os.makedirs(build_temp)
#         if not os.path.exists(build_lib):
#             os.makedirs(build_lib)

#         # configure cmake args
#         config = 'Debug' if self.debug else 'Release'
#         cmake_args = [
#             '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(build_lib),
#             '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
#             '-DCMAKE_BUILD_TYPE={}'.format(config)  # Not used on all platforms
#         ]

#         # call cmake
#         subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
#         subprocess.check_call(['cmake', '--build', '.'], cwd=build_temp)

#         # move back to the cwd
#         os.chdir(cwd)

# setup(
#     name='voxelizer',
#     version='0.1',
#     packages=find_packages(),
#     ext_modules=[CMakeExtension('voxelizer')],
#     cmdclass={
#         'build_ext': CMakeBuild,
#     }
# )