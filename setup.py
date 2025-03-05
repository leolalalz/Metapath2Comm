# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="Metapath2Comm",
    version="0.1.0",
    author="Xu Chenle",
    author_email="xuchenle@big.ac.cn",
    description="空间转录组异构网络分析工具包",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leolalalz/Metapath2Comm",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "scanpy>=1.8.0",
        "torch>=1.10.0",
        "torch-geometric>=2.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        # 添加其他依赖项...
    ],
    entry_points={
        "console_scripts": [
            "Metapath2Comm=Metapath2Comm.cli:main",
        ],
    },
)