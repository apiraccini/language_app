FROM python:3.10-slim-buster

COPY ./requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

WORKDIR /app
COPY ./app .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "420"]