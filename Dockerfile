FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
# Add ",rerank" to the extras for the cross-encoder (pulls in PyTorch: a much larger image).
ARG EXTRAS=openai,anthropic,ui
RUN pip install --no-cache-dir ".[${EXTRAS}]"

COPY corpus ./corpus
COPY evals ./evals

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "verified_doc_qa.api:app", "--host", "0.0.0.0", "--port", "8000"]
