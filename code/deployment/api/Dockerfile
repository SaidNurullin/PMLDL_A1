FROM python:3.9-slim-buster

# Install dependencies
RUN pip install fastapi uvicorn joblib numpy catboost
  
# Copy code and model to the container
WORKDIR /app
COPY . /app
  
# Define the entrypoint 
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
