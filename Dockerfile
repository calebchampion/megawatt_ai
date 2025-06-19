FROM python:3.13

WORKDIR /app

COPY requirements.txt .

# upgrade pip only
RUN pip install --upgrade pip

# run dependancies first before copying over the rest of the app
RUN pip install --no-cache-dir -r requirements.txt

# copy rest of app
COPY . .

# commands
CMD ["python", "deploy/run_chatbot.py"]