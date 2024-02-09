FROM python:3.12.1
COPY . /main
WORKDIR /main
RUN pip install -r requirements.txt
EXPOSE $PORT 
CMD Streamlit run main.py