FROM python:3.9-slim-buster

# Install Streamlit and dependencies
RUN pip install streamlit numpy

# Copy the Streamlit app code to the container
WORKDIR /app
COPY . /app

# Define the entrypoint (command to run when the container starts)
ENTRYPOINT ["streamlit", "run", "main.py"]
