FROM python:3.11-slim
WORKDIR /mnt/c/Users/zeiad/Downloads/Medical-LLM-Backend
COPY requirments.txt .
RUN pip install --no-cache-dir -r requirments.txt
COPY . .
EXPOSE 8080
CMD [ "python", "SingeltonModel.py" ]