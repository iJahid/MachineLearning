# rguILDQ7ccwhxWyTHfPgdsXHAjSGXDag
import requests

# api = 'https://api.windy.com/webcams/api/v3/webcams?countries=IN'
# api = 'https://api.windy.com/webcams/api/v3/webcams/1509611733'
api = 'https://api.windy.com/webcams/api/v3/webcams?lang=en&limit=10&offset=0&categoryOperation=and&sortKey=popularity&sortDirection=asc&include=images&continents=AF&categories=landscape'
# api = 'https://api.windy.com/webcams/api/v3/countries?lang=en'
headers = {
    'x-windy-api-key': 'rguILDQ7ccwhxWyTHfPgdsXHAjSGXDag'
}

response = requests.get(url=api, headers=headers)
print(response.json())
