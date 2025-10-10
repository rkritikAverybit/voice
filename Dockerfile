FROM python:3.12-bullseye

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip wheel setuptools
RUN pip install -r requirements.txt

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "10000"]
