FROM python:3.10-slim

WORKDIR /app

# Copy file requirements.txt dan install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project (termasuk app.py dan models/)
COPY . .

# Jalankan app
CMD ["python", "app.py"]
