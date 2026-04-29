# 1. Use a lightweight Python Linux image
FROM python:3.10-slim

# 2. Install system requirements for computer vision (C++ compilers and OpenCV tools)
RUN apt-get update && apt-get install -y git g++ libgl1 libglib2.0-0

# 3. Install PyTorch (CPU version for local Windows testing)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install Detectron2 directly from Facebook's GitHub
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 5. Set up the working directory inside the container
WORKDIR /app

# 6. Copy our requirements and install them
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 7. Copy all our files (including app.py and model_final.pth) into the container
COPY . .

# 8. Expose port 7860 for the web server
EXPOSE 7860

# 9. Start the Flask server
CMD ["python", "app.py"]