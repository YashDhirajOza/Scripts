import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Function to download a file
def download_file(url, directory):
    local_filename = os.path.join(directory, os.path.basename(urlparse(url).path))
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# Main function to scrape and download papers
def download_papers(url):
    # Make a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find all links in the page
    links = soup.find_all('a')
    
    # Directory to save downloaded files
    download_dir = 'papers'
    os.makedirs(download_dir, exist_ok=True)
    
    # Iterate through links to find papers
    for link in links:
        href = link.get('href')
        if href and href.endswith('.pdf'):  # Assuming papers are PDFs
            paper_url = urljoin(url, href)
            print(f"Downloading {paper_url}")
            try:
                download_file(paper_url, download_dir)
                print(f"Downloaded {paper_url}")
            except Exception as e:
                print(f"Failed to download {paper_url}: {str(e)}")

# Example usage:
if __name__ == "__main__":
    url = 'https://www.gtupaper.in/engineering/07/Computer/sem1/3110005/Basic-Electrical-Engineering'
    download_papers(url)
