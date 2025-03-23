# Use Python base image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 10000

# Start the application
CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]