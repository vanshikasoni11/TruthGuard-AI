from dotenv import load_dotenv
import os
import requests
load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
def search_news(query):

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        return []

    data = response.json()

    return data.get("articles", [])