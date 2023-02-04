FROM python:3.10.7-slim
MAINTAINER Damir Suleev "DaMIRka-Oo"

COPY ./ /mlops_suleev

WORKDIR /mlops_suleev/api_service

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get -y install curl
RUN apt-get install libgomp1

EXPOSE 5000

ENTRYPOINT ["python", "ml_api.py", "--host=0.0.0.0"]