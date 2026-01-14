FROM python:3.10

WORKDIR /app

# The full python image includes build dependencies like gcc, so we don't need to install them manually.
# If you prefer slim, you might need to debug apt-get limits or repository issues.

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
