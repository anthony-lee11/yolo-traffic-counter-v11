# Gunakan image Python resmi
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy semua file
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000
EXPOSE 5000

# Jalankan Flask app
CMD ["python", "app.py"]
