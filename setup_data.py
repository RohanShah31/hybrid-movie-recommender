import os
import urllib.request
import zipfile
import ssl

def download_movielens():
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(data_dir, "ml-1m.zip")
    
    print("Downloading MovieLens 1M dataset...")
    
    # Bypass SSL verification if certificates are missing
    ssl._create_default_https_context = ssl._create_unverified_context
    
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Zip file already exists.")
        
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
        
    print("Dataset ready in data/ml-1m/")

if __name__ == "__main__":
    download_movielens()
