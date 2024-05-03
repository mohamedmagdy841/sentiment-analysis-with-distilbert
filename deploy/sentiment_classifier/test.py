from fastapi import FastAPI
import  uvicorn
import requests

# app = FastAPI()
#
# @app.get('/')
# def index():
#     return {'message': 'Hello, World!'}
#
# @app.get('/{name}')
# def get_name(name: str):
#     return {'Welcome to fastapi': f'{name}'}

if __name__ == '__main__':
    url = "http://127.0.0.1:8000/predict"
    params = {"text": 'great'}

    response = requests.post(url, json=params)

    print((response.text))