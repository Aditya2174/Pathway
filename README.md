[tool.poetry]
name = "rag"
version = "0.1.0"
description = "list of dependies for running rag pipeline"
authors = ["Aditya Garg"]
readme = "README.md"

[tool.poetry.dependencies]
python = ">=3.10,<3.12"
pathway = {extras = ["xpack-llm"], version = "^0.15.2"}


[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
