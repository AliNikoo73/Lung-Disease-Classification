from setuptools import setup, find_packages

setup(
    name="lung-disease-classification",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.8.0",
        "tensorflow-addons>=0.16.1",
        "numpy>=1.19.5",
        "pandas>=1.3.5",
        "matplotlib>=3.5.1",
        "seaborn>=0.11.2",
        "scikit-learn>=1.0.2",
        "opencv-python>=4.5.5.64",
        "pillow>=8.4.0",
        "tqdm>=4.62.3",
        "grad-cam>=1.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A deep learning model for lung disease classification from chest X-ray images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lung-disease-classification",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
) 