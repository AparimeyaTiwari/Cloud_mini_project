FROM python:3.8-slim

WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app.py into the container
COPY app.py .

# Copy the util/ directory into the container
COPY util/ ./util/

# Copy the pages/ directory into the container
COPY pages/ ./pages/

# Copy the data_util.py file into the container
COPY data_util.py .

# Copy the data.csv file into the container
COPY data.csv .

# Copy the models/ directory into the container
COPY models/ ./models/

# Expose port 8501 for Streamlit
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
