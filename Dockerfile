# Use official Python base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create and set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install .

# Expose Shiny's default port
EXPOSE 8000

# Run the Shiny app
CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "8000", "app/app.py"]