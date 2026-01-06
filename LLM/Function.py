

def retrieve_similar_cities(city_index, cities, indices, k=20):
    neighbors = indices[city_index][1:k+1]
    results = []

    for idx in neighbors:
        city = cities[idx]
        text = (
            f"Ville : {city['name']}. "
            f"Pays : {city['country']}. "
            f"Population : {city['population']}. "
            f"Latitude : {city['lat']}, Longitude : {city['lon']}."
        )
        results.append(text)

    return results