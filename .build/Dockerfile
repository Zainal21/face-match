FROM python:3.9-alpine

LABEL maintener="Muhamad Zainal Arifin <muhammadzaindev@gmail.com>"

RUN apk add --no-cache \
    build-base \
    cmake \
    jpeg-dev \
    zlib-dev \
    openblas-dev \
    freetype-dev \
    libpng-dev \
    musl-dev \
    linux-headers \
    gfortran

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["flask", "run", "--host=0.0.0.0"]
