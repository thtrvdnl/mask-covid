FROM python:3.8-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ./req.txt req.txt
COPY . /app

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -qq -y \
    build-essential libpq-dev netcat libavdevice-dev libavfilter-dev libopus-dev libvpx-dev pkg-config libsrtp2-dev --no-install-recommends

RUN pip install --no-cache-dir -r req.txt

RUN adduser --disabled-password --gecos '' user

EXPOSE 8080

CMD ["gunicorn", "server:server_web", "--bind", "0.0.0.0:8080", "--worker-class", "aiohttp.GunicornWebWorker"]