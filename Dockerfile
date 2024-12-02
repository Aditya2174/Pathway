FROM python:3.11 AS base

RUN pip install poetry==1.8.4

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app
COPY ./pyproject.toml ./poetry.lock /app/
COPY ./PathwayPS /app/PathwayPS

RUN apt-get update
RUN apt-get install -y \
    build-essential \
    curl \
    libgl1-mesa-glx
RUN apt-get update
RUN apt install -y libprotobuf-dev protobuf-compiler
RUN apt-get update && apt-get -y install cmake

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN poetry run python -m pip install --upgrade pip
RUN poetry install
RUN poetry run python -m pip install torch
RUN poetry run python -m pip install -e "PathwayPS[all]"
RUN poetry install && rm -rf ${POETRY_CACHE_DIR}

COPY <<EOF ./temp.py
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
EOF
RUN poetry run python temp.py

FROM base_image AS server
ENV TESSDATA_PREFIX=/app
COPY ./data /app/data/
COPY ./src /app/src/
COPY ./eng.traineddata /app/
WORKDIR /app
EXPOSE 8756
ENTRYPOINT ["poetry", "run", "python", "src/server.py"]

FROM base_image AS pipeline
COPY ./src /app/src/
RUN chmod +x /app/src/wait-for.sh
WORKDIR /app
EXPOSE 8501
ENTRYPOINT ["/app/src/wait-for.sh", "poetry", "run", "python", "-m", "streamlit", "run", "src/pipeline.py"]
