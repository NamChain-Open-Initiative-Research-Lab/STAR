# Use PyTorch base image
FROM pytorch/pytorch:latest

# Create a non-root user for security
RUN useradd -m star-trainer
USER star-trainer

# Set working directory
WORKDIR /app

# Copy only training code (NOT dataset)
COPY src /app/src
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Set environment variables
ENV DATA_PATH="/data"
ENV MODEL_PATH="/models"

# Expose volumes (mount dataset + save models)
VOLUME ["/data", "/models"]

# Run training
CMD ["python", "src/train.py", "--data", "/data", "--save", "/models"]
