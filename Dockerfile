# Use an official Python runtime as the base image
FROM python:3.9-slim
# Set the working directory in the docker image
WORKDIR /app
# Copy the requirements.txt
COPY requirements.txt /app
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
# Env
ARG OPEN_AI_API_KEY
ENV OPEN_AI_API_KEY=$OPEN_AI_API_KEY
# Copy the current directory (on the host machine) to the docker image
COPY . /app
# Make port 8000 available to the world outside this container
EXPOSE 8000
# Container start command
CMD ["gunicorn", "main:app"]