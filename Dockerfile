# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# Set the working directory in the container to /app
WORKDIR /main

# Copy the current directory contents into the container at /app
COPY . /main

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt