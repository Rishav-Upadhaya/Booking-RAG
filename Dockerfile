FROM python:3.12-slim

WORKDIR /app

# Dependencies first: this layer only rebuilds when requirements.txt changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code last: code changes reuse the cached dependency layer.
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
COPY pyproject.toml ./
RUN pip install --no-cache-dir --no-deps .

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]