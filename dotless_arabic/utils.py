import os

import requests


def download_file(file_url, file_name, output_dir):
    outfile = os.path.join(output_dir, file_name)
    response = requests.get(
        file_url,
        stream=True,
    )
    with open(outfile, "wb") as output:
        output.write(response.content)
    return True
