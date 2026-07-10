from dotenv import load_dotenv
import os
import requests
load_dotenv()


API_KEY = os.getenv("GOOGLE_FACT_CHECK_API_KEY")

def fact_check(query):

    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={API_KEY}"

    response = requests.get(url)
    data = response.json()

    claims = data.get("claims", [])

    if not claims:
        return None

    results = []

    for c in claims[:3]:
        review = c.get("claimReview", [{}])[0]
        results.append({
            "text": c.get("text"),
            "rating": review.get("textualRating"),
            "publisher": review.get("publisher", {}).get("name")
        })

    return results