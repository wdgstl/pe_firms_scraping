import os
import time
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Keywords to skip crawling (unchanged)
EXCLUDE_KEYWORDS = {
    "careers", "career", "jobs", "team", "people",
    "privacy", "terms", "legal",
    "contact", "support", "help", "faq",
    "blog", "news", "press", "media",
    "events", "webinar", "culture", "advisor", "advisors",
    "login", "signup", "register", "subscribe",
    "cookie", "rss", "sitemap",
    "leadership", ".pdf", ".jpg", ".jpeg",
    "branch", ".xlsx", "email", "article", "report", ".mp4", ".mp3"
}

# Ensure output directory exists
OUTPUT_DIR = "scraped_pages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize embedding model
# model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def crawl_site(start_url, max_pages=1000):
    """
    Crawl internal pages up to max_pages, extract all headings and paragraphs
    in the order they appear (writing header text only),
    and dump them to a single .txt file with page breaks.
    """
    visited, to_visit = set(), [start_url]
    base_domain = urlparse(start_url).netloc.replace("www.", "")
    firm_name = base_domain.replace('.', '_')
    output_file = os.path.join(OUTPUT_DIR, f"{firm_name}.txt")

    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--disable-gpu")
    opts.add_experimental_option("prefs", {
        "profile.managed_default_content_settings.images": 2,
        "profile.managed_default_content_settings.stylesheets": 2,
        "profile.managed_default_content_settings.fonts": 2,
    })
    driver = webdriver.Chrome(options=opts)

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print("Visiting:", url)
        driver.get(url.split('#')[0])

        try:
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except:
            pass

        soup = BeautifulSoup(driver.page_source, "html.parser")
        with open(output_file, "a", encoding="utf-8") as f:
            for el in soup.find_all(["h1","h2","h3","h4","h5","h6","p"]):
                text = el.get_text(separator=" ", strip=True)
                if not text:
                    continue
                if el.name.startswith("h"):
                    f.write(f"\n{text}\n")
                else:
                    f.write(text + "\n")
            f.write("\n---PAGE BREAK---\n\n")

        for tag in soup.find_all("a", href=True):
            href = urljoin(url, tag["href"])
            p2 = urlparse(href)
            root_link = p2._replace(query="", fragment="").geturl()
            if p2.netloc.replace("www.", "") != base_domain:
                continue
            if any(kw in root_link.lower() for kw in EXCLUDE_KEYWORDS):
                continue
            candidate = root_link + (f"#{p2.fragment}" if p2.fragment else "")
            if candidate not in visited and candidate not in to_visit:
                to_visit.append(candidate)

    driver.quit()
    return output_file


def chunk_text(lines):
    """
    Split text into chunks by single blank-line separators, but ensure that headers
    stay with their following paragraphs. Page-break markers also split.
    Returns a list of chunk strings.
    """
    chunks, current = [], []
    for line in lines:
        stripped = line.strip()
        if stripped == "---PAGE BREAK---":
            if current:
                chunks.append("\n".join(current).strip())
                current = []
            continue
        if stripped == "":
            if current and len(current) > 1:
                chunks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        chunks.append("\n".join(current).strip())
    return chunks


def embed_and_rank_paragraphs_thesis(paragraphs, query, industry, top_k=5,
                                    min_words=3, min_chars=30, boost_weight=0.2):
    """
    Rank text chunks by semantic similarity to a thesis-focused query,
    with an optional industry keyword boost. Returns up to top_k (chunk, score).
    """
    import re

    # Simple noise filter (reuse logic)
    def is_noise(p):
        if len(p.split()) < min_words or len(p) < min_chars:
            return True
        letters = [c for c in p if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
            return True
        return False

    clean = [p for p in paragraphs if not is_noise(p)] or paragraphs

    # Compute embeddings and cosine similarities
    qv   = model.encode(query, convert_to_numpy=True)
    embs = model.encode(clean, batch_size=64, convert_to_numpy=True)
    sims = np.dot(embs, qv) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(qv))

    # Industry keyword boost
    ind_flags = np.array([
        1 if industry.lower() in p.lower() else 0
        for p in clean
    ])

    # Combine scores
    scores = sims + boost_weight * ind_flags

    # Select top_k
    idxs = np.argsort(scores)[::-1][:top_k]
    return [(clean[i], float(scores[i])) for i in idxs]


def embed_and_rank_paragraphs(paragraphs, query, top_k=10,
                              min_words=5, min_chars=60, boost_weight=0.2):
    """
    Embed+rank text blocks by similarity, boosting those containing
    an expanded set of PE-industry keywords. Returns list of (chunk, score).
    """
    KEYWORDS = {
        "focus", "invest", "investment", "strategy", "portfolio", "sector", "thesis",
        "acquire", "acquires", "grows", "grow", "business", "company",
        "holding", "model", "mission", "goal",
        "healthcare", "medtech", "medical devices", "pharmaceuticals", "biotech",
        "technology", "software", "cloud computing", "SaaS", "AI", "machine learning",
        "cybersecurity", "blockchain", "fintech", "insurtech",
        "energy", "renewable energy", "oil & gas", "utilities",
        "industrial", "manufacturing", "automotive", "automotive components",
        "transportation", "logistics", "supply chain",
        "consumer", "consumer goods", "FMCG", "ecommerce", "retail",
        "food & beverage", "hospitality", "travel", "tourism",
        "education", "edtech",
        "media", "digital media", "streaming", "gaming",
        "telecommunications", "5G", "IoT", "internet of things",
        "real estate", "infrastructure", "construction",
        "financial services", "banking", "insurance", "wealth management",
        "mining", "metals", "chemicals",
        "advertising", "adtech", "martech",
        "HR tech", "human resources",
        "data centers", "cloud infrastructure", "hvac", "construction"
    }

    def is_noise(p):
        if len(p.split()) < min_words or len(p) < min_chars:
            return True
        letters = [c for c in p if c.isalpha()]
        if letters and sum(1 for c in letters if c.isupper())/len(letters) > 0.6:
            return True
        return False

    clean = [p for p in paragraphs if not is_noise(p)] or paragraphs
    qv = model.encode(query, convert_to_numpy=True)
    embs = model.encode(clean, convert_to_numpy=True, batch_size=64)
    sims = np.dot(embs, qv) / (np.linalg.norm(embs, axis=1) * np.linalg.norm(qv))
    flags = np.array([1 if any(kw in p.lower() for kw in KEYWORDS) else 0 for p in clean])
    scores = sims + boost_weight * flags
    idxs = np.argsort(scores)[::-1][:top_k]
    return [(clean[i], float(scores[i])) for i in idxs]


def read_txt(folder_path, filename):
    full_path = os.path.join(folder_path, filename)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read().splitlines()


def delete_txt(folder_path, filename):
    file_path = os.path.join(folder_path, filename)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
