import requests
import zipfile
import pathlib

def download_data():
    data_path = pathlib.Path("data")
    train_dir = data_path / "pizza_steak_sushi" / "train"
    test_dir = data_path / "pizza_steak_sushi" / "test"

    if train_dir.is_dir():
        print("Data already exists, skipping download.")
        return train_dir, test_dir

    print("Downloading data...")
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    r = requests.get(url)
    zip_path = data_path / "pizza_steak_sushi.zip"
    open(zip_path, "wb").write(r.content)

    print("Extracting data...")
    zipfile.ZipFile(zip_path).extractall(data_path)
    print("Done.")

    return train_dir,test_dir

if __name__=='__main__':
    download_data()