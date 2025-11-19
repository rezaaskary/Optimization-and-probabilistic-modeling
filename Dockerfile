FROM python:3.9
MAINTAINER  Reza Askari <reza.askary99@gmail.com>

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "unittest", "-s", "Tests"]