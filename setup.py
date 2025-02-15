from setuptools import setup, find_packages

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

