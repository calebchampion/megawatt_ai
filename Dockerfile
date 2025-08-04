FROM python:3.13.3

WORKDIR /AI_MegawattS

COPY requirements.txt .
#run dependancies first before copying over the rest of the app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#copy rest of app
COPY . ./

#commands
EXPOSE 8501
CMD ["streamlit", "run", "AI_Megawatt/app.py", "--server.port", "8501"] 