FROM python:3.11-buster

RUN pip install poetry==1.8.4

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./
COPY PathwayPS ./PathwayPS
COPY src ./src
COPY data ./data
RUN touch README.md

RUN poetry install --without pathway && rm -rf $POETRY_CACHE_DIR
RUN pip install -e PathwayPS
EXPOSE 8756

ENTRYPOINT ["poetry", "run", "python", "-m", "src/server.py"]