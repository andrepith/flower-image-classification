import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm
import platformdirs

load_dotenv()

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

def get_cache_dir():
    return Path(platformdirs.user_cache_dir("ml_fastapi_flower"))

def download_dataset(dataset_name="alxmamaev/flowers-recognition", extract=True):
    os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = KAGGLE_KEY

    api = KaggleApi()
    api._load_config(config_data={
        "username": KAGGLE_USERNAME,
        "key": KAGGLE_KEY
    })
    api.authenticate()

    cache_dir = get_cache_dir()
    zip_path = cache_dir / "flowers.zip"
    dataset_dir = cache_dir / "flowers"

    if dataset_dir.exists():
        print(f"✅ Dataset already exists in cache: {dataset_dir}")
        return dataset_dir

    print(f"📦 Downloading dataset to: {zip_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download (quiet=False shows download progress from kaggle lib itself)
    api.dataset_download_files(dataset_name, path=cache_dir, unzip=False, quiet=False)

    # Locate the actual downloaded zip
    downloaded_zip = next(cache_dir.glob("*.zip"), None)
    if downloaded_zip is None or not downloaded_zip.exists():
        raise FileNotFoundError(f"❌ Expected a .zip file in {cache_dir}, but none found.")
    print(f"📦 Found ZIP: {downloaded_zip}")

    # Show progress while extracting
    print("📂 Extracting dataset...")
    with zipfile.ZipFile(downloaded_zip, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        with tqdm(total=total_files, unit="file", desc="📦 Extracting") as pbar:
            for member in zip_ref.infolist():
                zip_ref.extract(member, path=cache_dir)
                pbar.update(1)

    downloaded_zip.unlink(missing_ok=True)  # delete zip file after extraction

    # Show progress while extracting
    print("📂 Extracting dataset...")
    with zipfile.ZipFile(downloaded_zip, 'r') as zip_ref:
        total_files = len(zip_ref.infolist())
        with tqdm(total=total_files, unit="file", desc="📦 Extracting") as pbar:
            for member in zip_ref.infolist():
                zip_ref.extract(member, path=cache_dir)
                pbar.update(1)

    # Cleanup ZIP
    downloaded_zip.unlink(missing_ok=True)

    # Find actual extracted folder
    for subdir in cache_dir.iterdir():
        if subdir.is_dir() and "flower" in subdir.name.lower():
            if subdir.name != "flowers":
                subdir.rename(dataset_dir)
            break

    print(f"✅ Dataset ready at: {dataset_dir}")
    return dataset_dir
