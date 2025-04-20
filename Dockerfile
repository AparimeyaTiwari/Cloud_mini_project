# Use a lightweight Python image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files and folders
COPY app.py .
COPY data.csv .
COPY data_util.py .
COPY pages/ ./pages/
COPY util/ ./util/
COPY models/ ./models/

# Expose the port Streamlit uses
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


#for dockers files