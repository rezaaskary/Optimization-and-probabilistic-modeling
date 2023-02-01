FROM python:3.9
ADD main.py
RUN pip install -r requirements.txt
CMD ['python', '-m', 'unittest', '-s','Tests']