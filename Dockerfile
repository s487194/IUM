
FROM python:3.8-slim-buster


WORKDIR /


RUN apt-get update && apt-get install -y \
    figlet \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip



RUN pip install torch pandas numpy scikit-learn kaggle mlflow


COPY . /






