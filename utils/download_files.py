import requests
import os
from tqdm import tqdm

# Download process logs json file from zenodo.
def download_json_from_zenodo(record_id, output_dir='/data'):
    """
    Downloads only .json files from a specific Zenodo record.
    """
    api_url = f"https://zenodo.org/api/records/{record_id}"
   
    try:
        r = requests.get(api_url)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metadata: {e}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Scanning record: {data['metadata']['title']}")
   
    files_found = 0
   
    for file_info in data.get('files', []):
        file_name = file_info['key']
       
        # Check if the file ends with .json (case-insensitive)
        if not file_name.lower().endswith('.json'):
            continue
           
        files_found += 1
        download_url = file_info['links']['self']
        file_size = file_info['size']
        output_path = os.path.join(output_dir, file_name)
       
        if os.path.exists(output_path):
            print(f"File {file_name} already exists. Skipping.")
            continue

        print(f"Downloading {file_name}...")
       
        with requests.get(download_url, stream=True) as file_r:
            file_r.raise_for_status()
            with open(output_path, "wb") as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name) as pbar:
                    for chunk in file_r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                           
    if files_found == 0:
        print("No .json files found in this record.")
    else:
        print(f"Downloaded {files_found} .json file(s) to {output_dir}.")