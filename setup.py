from setuptools import setup, find_packages
import sys

print('locate packages', find_packages())
print(sys.path)

import site
print(site.getusersitepackages())
site.addsitedir(site.getusersitepackages())
print(sys.path)

setup(
    name='xlmlib',
    version='0.1.0',
    author='yelong shen',
    author_email='shengyelong@gmail.com',
    description='A simple library with x model.',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

