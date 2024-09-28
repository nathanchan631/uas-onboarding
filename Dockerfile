FROM python:3.10-slim

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY src/requirements.txt /src/
RUN pip3 install -r /src/requirements.txt

COPY . /app
WORKDIR /app

CMD ["./src/start_gunicorn.sh"]