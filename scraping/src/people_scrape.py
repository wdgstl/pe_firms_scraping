import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from collections import deque
import csv, json, re, sys
from typing import List, Dict, Tuple, Set, Optional

LEADERSHIP_KEYWORDS = [
    'team', 'leadership', 'people', 'about', 'management', 'our-story', 'who-we-are', 'partners'
]
NAME_PATTERN = re.compile(r'^[A-Z][a-zA-Z\-\.]+(?: [A-Z][a-zA-Z\-\.]+){1,3}$')
EXCLUDE_NAME_PREFIXES = {
    # Common site navigation & about sections
    'Our', 'About', 'Leadership', 'How', 'The', 'Who', 'What', 'Why', 'Where', 'When',
    'Contact', 'Careers', 'Team', 'Partners', 'People', 'Staff', 'Management', 'Board', 'Advisors',
    
    # Marketing / culture slogans
    'Culture', 'Values', 'Mission', 'Vision', 'Story', 'Journey', 'History', 'Legacy', 'Commitment',
    'Promise', 'Innovation', 'Excellence', 'Sustainability', 'Diversity', 'Inclusion',
    'Integrity', 'Transparency', 'Trust', 'Impact', 'Passion', 'Collaboration',
    
    # Generic business fluff
    'Solutions', 'Services', 'Products', 'Capabilities', 'Expertise', 'Experience', 'Industries',
    'Markets', 'Opportunities', 'Strategy', 'Insights', 'Resources', 'Events',
    
    # Action / CTA words
    'Make', 'Create', 'Build', 'Explore', 'Discover', 'Learn', 'Read', 'Watch', 'Join', 'Sign',
    'Register', 'Apply', 'Donate', 'Help', 'Support', 'Invest', 'Partner', 'Grow', 'Launch',
    
    # Tagline / headline words
    'Long-Term', 'Transparent', 'Founding', 'Global', 'Worldwide', 'National', 'Regional',
    'Local', 'Proud', 'Award-winning', 'Recognized', 'Certified',
    
    # Numbers or fake headers
    '5,000', '2024', '2023', '2022', '50th', '100th',
    
    # We/our statements
    'We', "We're", "We’re", 'Us', 'Ourselves',
    
    # Miscellaneous noise
    'Home', 'News', 'Press', 'Media', 'Blog', 'Article', 'Case', 'FAQ', 'Helpdesk', 'Portal',
    'Dashboard', 'Account', 'Login', 'Register'
}
CONTAINER_KEYWORDS = ['profile','team-member','member','bio','leadership','person','staff','employee']
TITLE_HINTS = ['title','role','position','job','designation']
MIN_BIO_LENGTH = 50
REQUEST_TIMEOUT = 8
HEADERS = {"User-Agent": "Mozilla/5.0 (bio-scraper/1.1)"}
SKIP_EXTS = {'.pdf','.jpg','.jpeg','.png','.gif','.svg','.webp','.mp4','.mov','.zip','.doc','.docx','.xls','.xlsx'}

def _norm(url: str) -> str:
    """Drop query/fragment; normalize trailing slash."""
    p = urlparse(url)
    if any(p.path.lower().endswith(ext) for ext in SKIP_EXTS):
        return ""
    norm = urlunparse((p.scheme, p.netloc, p.path.rstrip('/'), "", "", "")) or url
    return norm

def _looks_like_leadership_path(path: str) -> bool:
    p = path.lower()
    return any(k in p for k in LEADERSHIP_KEYWORDS)

def _anchor_text_says_team(a: str) -> bool:
    t = (a or "").strip().lower()
    return any(k in t for k in ['team','leadership','people','about','partners','management'])

class LeadershipCrawler:
    def __init__(self, start_url: str, max_pages: int = 300, max_depth: int = 3, verbose: bool = True):
        self.start_url = start_url.rstrip("/")
        self.domain = urlparse(self.start_url).netloc
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.verbose = verbose
        self.visited: Set[str] = set()
        self.seen: Set[Tuple[str, str]] = set()  # (name, page_url)
        self.results: List[Dict[str, str]] = []

        # session with retries
        self.sess = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        self.sess.mount("http://", HTTPAdapter(max_retries=retries))
        self.sess.mount("https://", HTTPAdapter(max_retries=retries))

    def crawl(self) -> List[Dict[str, str]]:
        q = deque([(self.start_url, 0)])
        while q and len(self.visited) < self.max_pages:
            url, depth = q.popleft()
            url = _norm(url)
            if not url or url in self.visited:
                continue
            self.visited.add(url)
            if self.verbose:
                print(f"[{len(self.visited)}/{self.max_pages}] depth={depth} → {url}")

            try:
                res = self.sess.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
                ctype = res.headers.get('Content-Type', '')
                if res.status_code != 200 or 'text/html' not in ctype:
                    if self.verbose:
                        print(f"  skip: status={res.status_code}, type={ctype}")
                    continue
                soup = BeautifulSoup(res.text, 'html.parser')
            except Exception as e:
                if self.verbose:
                    print(f"  fail: {e}")
                continue

            # Only parse leadership-like pages
            if _looks_like_leadership_path(urlparse(url).path):
                for rec in self._extract_profiles(soup, source_url=url):
                    key = (rec["name"], rec["source_url"])
                    if key in self.seen:
                        continue
                    self.seen.add(key)
                    self.results.append(rec)

            # Enqueue next links — but restrict aggressively
            if depth < self.max_depth:
                for a in soup.find_all('a', href=True):
                    abs_url = urljoin(url, a['href'])
                    if urlparse(abs_url).netloc != self.domain:
                        continue
                    # Only enqueue if the link *looks* like a leadership page OR anchor text suggests it
                    u_norm = _norm(abs_url)
                    if not u_norm:
                        continue
                    path = urlparse(u_norm).path
                    if _looks_like_leadership_path(path) or _anchor_text_says_team(a.get_text()):
                        if u_norm not in self.visited:
                            q.append((u_norm, depth + 1))
        return self.results

    def save_csv(self, filepath: str) -> None:
        fieldnames = ["name", "position", "bio", "source_url"]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self.results:
                w.writerow(r)

    def save_jsonl(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -------- extraction helpers --------
    def _extract_profiles(self, soup: BeautifulSoup, source_url: str) -> List[Dict[str, str]]:
        records: List[Dict[str, str]] = []
        containers = soup.find_all(
            lambda tag: tag.name in ['div', 'section', 'article']
            and any(kw in ' '.join(tag.get('class', [])).lower() for kw in CONTAINER_KEYWORDS)
        )
        if not containers:
            containers = soup.find_all(['div', 'section', 'article'])

        for c in containers:
            heading = None
            for tagname in ['h1','h2','h3','h4']:
                heading = c.find(tagname)
                if heading: break
            if not heading:
                continue

            name = heading.get_text(" ", strip=True)
            if not name or any(ex.lower() in name.lower() for ex in EXCLUDE_NAME_PREFIXES) or not NAME_PATTERN.match(name):
                continue


            position = self._extract_title_nearby(heading) or self._extract_title_by_class(c)

            full_text = c.get_text(" ", strip=True)
            bio_text = full_text
            if full_text.startswith(name):
                bio_text = full_text[len(name):].strip()
            if position and bio_text.startswith(position):
                bio_text = bio_text[len(position):].strip()

            if len(bio_text) < MIN_BIO_LENGTH:
                alt_bio = self._paragraphs_after(heading, limit_chars=800)
                if not alt_bio or len(alt_bio) < MIN_BIO_LENGTH:
                    continue
                bio_text = alt_bio

            records.append({
                "name": name,
                "position": position or "",
                "bio": self._clean_spaces(bio_text),
                "source_url": source_url
            })
        if self.verbose:
            print(f"  extracted {len(records)} profiles from {source_url}")
        return records

    def _extract_title_nearby(self, heading) -> Optional[str]:
        for sib in heading.find_all_next(limit=4):
            if sib.name in ['h5','h6','p','span','small','div']:
                text = sib.get_text(" ", strip=True)
                if self._looks_like_title(text):
                    return text
        return None

    def _extract_title_by_class(self, container) -> Optional[str]:
        for kw in TITLE_HINTS:
            el = container.find(attrs={"class": re.compile(kw, re.I)})
            if el:
                txt = el.get_text(" ", strip=True)
                if self._looks_like_title(txt):
                    return txt
        return None

    def _looks_like_title(self, text: str) -> bool:
        if not text or len(text) > 80:
            return False
        role_words = ['partner','principal','associate','manager','managing','director','vp','vice','president',
                      'analyst','founder','co-founder','chief','officer','cto','ceo','cfo','coo','chair']
        t = text.lower()
        return any(w in t for w in role_words) or ("," in text and len(text) <= 60)

    def _paragraphs_after(self, heading, limit_chars: int = 600) -> str:
        texts = []
        p = heading.find_next('p')
        while p and sum(len(x) for x in texts) < limit_chars:
            txt = p.get_text(" ", strip=True)
            if txt:
                texts.append(txt)
            p = p.find_next_sibling('p')
        return " ".join(texts).strip()

    def _clean_spaces(self, s: str) -> str:
        return re.sub(r'\s+', ' ', s).strip()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python leadership_crawler.py <start_url> <output.csv>")
        sys.exit(1)

    start_url = sys.argv[1]
    out_path = sys.argv[2]

    crawler = LeadershipCrawler(start_url, max_pages=200, max_depth=3, verbose=True)
    results = crawler.crawl()
    print(f"Found {len(results)} bios")

    if out_path.lower().endswith(".jsonl"):
        crawler.save_jsonl(out_path)
    else:
        crawler.save_csv(out_path)

    print(f"Wrote {len(results)} records to {out_path}")
