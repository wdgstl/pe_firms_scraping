import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re

# Pattern to identify leadership/profile links
PROFILE_LINK_PATTERN = re.compile(r"/(team|people|leadership|about#team)/?", re.IGNORECASE)

# Regex for matching person name in link text or heading
NAME_PATTERN = re.compile(r'^[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}$')


def get_profile_links(start_url):
    """
    Fetch the start_url page and return a set of absolute URLs
    for links whose href matches PROFILE_LINK_PATTERN.
    """
    try:
        res = requests.get(start_url, timeout=5)
        res.raise_for_status()
    except Exception as e:
        print(f"Error fetching {start_url}: {e}")
        return set()

    soup = BeautifulSoup(res.text, 'html.parser')
    domain = urlparse(start_url).netloc
    links = set()

    for a in soup.find_all('a', href=True):
        href = a['href']
        # Identify potential profile links by URL pattern
        if PROFILE_LINK_PATTERN.search(href):
            abs_url = urljoin(start_url, href)
            # Only include same-domain links
            if urlparse(abs_url).netloc == domain:
                links.add(abs_url)

    return links


def extract_name(profile_url):
    """
    Fetch profile_url and extract the person's name from the first heading.
    """
    try:
        res = requests.get(profile_url, timeout=5)
        res.raise_for_status()
    except Exception as e:
        print(f"Error fetching {profile_url}: {e}")
        return None

    soup = BeautifulSoup(res.text, 'html.parser')
    # Look for <h1> or <h2> as name
    heading = soup.find(['h1', 'h2'])
    if heading:
        name = heading.get_text(strip=True)
        if NAME_PATTERN.match(name):
            return name
    # Fallback: use link path segment as name
    path = urlparse(profile_url).path.rstrip('/').split('/')[-1]
    return path.replace('-', ' ').title()


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python people_scrape.py <start_url>")
        sys.exit(1)

    start_url = sys.argv[1]
    profile_links = get_profile_links(start_url)
    print(f"Found {len(profile_links)} potential profile pages.")

    names = set()
    for url in profile_links:
        name = extract_name(url)
        if name:
            names.add(name)

    print("\nLeadership Names:\n")
    for name in sorted(names):
        print(name)
