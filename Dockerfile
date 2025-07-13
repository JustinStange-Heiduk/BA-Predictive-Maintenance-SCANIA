# TensorFlow GPU Container (NVIDIA NGC 25.02, CUDA 12.8, Python 3.12)
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Set working directory
WORKDIR /workspace

# Copy dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source code
COPY ./src ./src
COPY ./notebooks ./notebooks

# Default command
CMD ["python", "./src/main.py"]
