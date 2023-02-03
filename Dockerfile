FROM python:3.9
MAINTAINER Mohammad Reza Askari <maskari@hawk.iit.edu>
RUN pip install -r requirements.txt
CMD ['python', '-m', 'unittest', '-s','Tests']