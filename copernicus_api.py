import requests
import getpass
import pandas as pd


def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()  # error bad response
    except Exception as e:
        raise Exception(f"Access token creation failed. Response from the server was: {r.json()}")

    return r.json()["access_token"]



# Example usage: Secure password input
username = getpass.getpass("Enter your username: ")
password = getpass.getpass("Enter your password: ")
access_token = get_access_token(username, password)

start_date = "2016-01-01"
end_date = "2025-01-01"
aoi = "POLYGON((24.11269 44.049853,24.057758 44.061569,24.042051 44.041403,24.102304 44.031657,24.11269 44.049853))"
top_value = "80"
cloud_cover_threshold = "10.00"
#top default e setat pe 20 daca nu pun nimic dar imi da EROARE daca il pun prea mare deci trebuie in jur de 100/200/300
search_url = f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=Collection/Name eq 'SENTINEL-2' and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud_cover_threshold}) and ContentDate/Start gt {start_date} and ContentDate/Start lt {end_date} and OData.CSC.Intersects(area=geography'SRID=4326;{aoi}')&$top={top_value}"

headers = {"Authorization": f"Bearer {access_token}"}

response = requests.get(search_url, headers=headers)

if response.status_code == 200 and response.json().get('value'):
    json_response = response.json()
    df = pd.DataFrame.from_dict(json_response['value'])

    if df.empty:
        print("Data frame empty.")
    else:
        # Iterate through each product and download it
        for _, product in df.iterrows():
            product_id = product['Id']
            product_name = product['Name']

            print(f"Selected product: {product_name}")

            # Download URL
            download_url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"

            session = requests.Session()
            session.headers.update({"Authorization": f"Bearer {access_token}"})

            download_response = session.get(download_url, stream=True)
            print(f"Download status code for {product_name}: {download_response.status_code}")
            print("Content-Type:", download_response.headers.get("Content-Type"))

            with open(f"{product_name}.zip", "wb") as file:
                for chunk in download_response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

            print(f"Download complete for {product_name}.")
else:
    print("No products found matching the criteria but handled okay.")