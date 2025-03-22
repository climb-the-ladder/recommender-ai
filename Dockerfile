FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add Gunicorn for production deployment
RUN pip install gunicorn

COPY . .

EXPOSE 5001

# Use environment variable to determine whether to run in development or production mode
ENV FLASK_ENV=production

# Use Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"] 
