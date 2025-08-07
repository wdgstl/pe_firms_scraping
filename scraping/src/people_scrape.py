import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import re

# Keywords to identify leadership-related URLs (path filtering)
LEADERSHIP_KEYWORDS = [
    'team', 'leadership', 'people', 'about', 'management', 'our-story', 'who-we-are'
]

# Regex to match person name patterns: two to four capitalized words
NAME_PATTERN = re.compile(r'^[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}$')

# Exclude common non-person headings
EXCLUDE_NAME_PREFIXES = {
    'Our', 'Make', 'Culture', 'Long-Term', 'Transparent', 'Sign', 'Garden', 
    'We', "We're", 'Founding', '5,000'
}

# Classes indicating a profile container
CONTAINER_KEYWORDS = ['profile', 'team-member', 'member', 'bio', 'leadership']

# Minimum length for bio text (after name removal)
MIN_BIO_LENGTH = 50


def extract_profiles(soup):
    """
    Find all profile containers and extract (name, bio) pairs with full container text.
    """
    profiles = []
    containers = soup.find_all(lambda tag: tag.name in ['div', 'section'] \
                                 and any(kw in ' '.join(tag.get('class', [])) for kw in CONTAINER_KEYWORDS))
    for container in containers:
        heading = container.find(['h1', 'h2', 'h3'])
        if not heading:
            continue
        name = heading.get_text(strip=True)
        if not NAME_PATTERN.match(name) or name.split()[0] in EXCLUDE_NAME_PREFIXES:
            continue
        # Get full text of container
        full_text = container.get_text(separator=' ', strip=True)
        # Remove name from the beginning
        bio_text = full_text[len(name):].strip() if full_text.startswith(name) else full_text
        if len(bio_text) < MIN_BIO_LENGTH:
            continue
        profiles.append((name, bio_text))
    return profiles


def crawl_leadership(start_url, max_pages=500):
    visited = set()
    queue = deque([start_url])
    domain = urlparse(start_url).netloc
    bios = []
    seen_names = set()

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        print(f"Crawling {url}")
        try:
            res = requests.get(url, timeout=5)
            if res.status_code != 200 or 'text/html' not in res.headers.get('Content-Type', ''):
                continue
            soup = BeautifulSoup(res.text, 'html.parser')
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            continue

        # Enqueue internal links
        for a in soup.find_all('a', href=True):
            abs_url = urljoin(start_url, a['href'])
            if urlparse(abs_url).netloc == domain and abs_url not in visited:
                queue.append(abs_url)

        # Only crawl pages with leadership-related paths
        if not any(keyword in urlparse(url).path.lower() for keyword in LEADERSHIP_KEYWORDS):
            continue

        # Extract full-profile entries
        for name, bio in extract_profiles(soup):
            if name in seen_names:
                continue
            bios.append((name, bio))
            seen_names.add(name)

    return bios


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python leadership_crawler.py <start_url>")
        sys.exit(1)

    start_url = sys.argv[1]
    results = crawl_leadership(start_url)
    print("\nFound leadership bios:\n")
    for name, bio in results:
        print(f"Name: {name}\nBio: {bio}\n{'-'*60}")
