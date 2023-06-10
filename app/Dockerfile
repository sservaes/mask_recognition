# Use the official Python base image
FROM python:3.11.2-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Expose the default Streamlit port
EXPOSE 850

# Run the Streamlit app
CMD streamlit run app.py \
    --server.port=8501