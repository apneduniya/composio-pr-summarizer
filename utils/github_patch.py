import requests


def get_patch(url: str) -> str:
    response = requests.get(url)
    return response.text



