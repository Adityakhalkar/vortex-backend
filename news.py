import requests

def get_natural_calamity_news(api_key, location="India", query="Latest news related to natural clamity in India"):
    # Set up the parameters for the API request
    params = {
        "q": query,
        "location": location,
        "tbm": "nws",  # Fetches news results
        "api_key": api_key
    }

    # Send the request to SerpApi
    response = requests.get("https://serpapi.com/search.json", params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Extract news articles from the response
        news_articles = data.get('news_results', [])

        # Print the headlines, description, image, and link of the news articles
        for index, article in enumerate(news_articles):
            title = article.get('title')
            link = article.get('link')
            snippet = article.get('snippet', 'No description available')
            thumbnail = article.get('thumbnail', 'No image available')

            print(f"{index + 1}. {title}")
            print(f"Description: {snippet}")
            print(f"Link: {link}")
            print(f"Image: {thumbnail}\n")

    else:
        print(f"Failed to fetch news: {response.status_code}")

# Replace 'your_api_key' with your actual SerpApi key
api_key = "ef4e7a7cd7387a227216587851e43fe41528b573d7cc2405d1e90f9190ff082f"

# Fetch natural calamity news in India
get_natural_calamity_news(api_key)