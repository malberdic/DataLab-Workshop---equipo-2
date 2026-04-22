
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
WORKDIR /app/src/API
CMD ["uvicorn", "review_api:app", "--host", "0.0.0.0", "--port", "7860"]