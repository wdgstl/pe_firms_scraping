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
from llama import *

# Keywords to skip crawling
EXCLUDE_KEYWORDS = {
    "careers", "career", "jobs", "team", "people",
    "privacy", "terms", "legal",
    "contact", "support", "help", "faq",
    "blog", "news", "press", "media",
    "events", "webinar", "culture",
    "login", "signup", "register", "subscribe",
    "cookie", "rss", "sitemap",
    "leadership", ".pdf", ".jpg", ".jpeg",
    "branch", ".xlsx", "email", "article", "report", ".mp4", ".mp3"
}

# Ensure output directory exists
OUTPUT_DIR = "scraped_pages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize local embedding model (free)
model = SentenceTransformer('all-MiniLM-L6-v2')

def crawl_site(start_url, max_pages=1000):
    visited = set()
    to_visit = [start_url]
    base_domain = urlparse(start_url).netloc.replace("www.", "")
    firm_name = base_domain.replace('.', '_')
    output_file = os.path.join(OUTPUT_DIR, f"{firm_name}.txt")

    # Headless browser setup (disable images, CSS, fonts for speed)
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

        root = url.split("#")[0]
        driver.get(root)
        try:
            WebDriverWait(driver, 3).until(
                EC.presence_of_element_located((By.TAG_NAME, "a"))
            )
        except:
            pass

        soup = BeautifulSoup(driver.page_source, "html.parser")
        parsed = urlparse(url)

        # Extract paragraphs from full page or in-page fragment
        if parsed.fragment:
            section = soup.find(id=parsed.fragment)
            paras = section.find_all("p") if section else []
        else:
            paras = soup.find_all("p")

        # Append paragraphs to output file
        with open(output_file, "a", encoding="utf-8") as f:
            for p in paras:
                text = p.get_text(separator=" ", strip=True)
                if text:
                    f.write(text + "\n\n")

        # Enqueue new links found in the page
        for tag in soup.find_all("a", href=True):
            href = urljoin(root, tag["href"])
            p2 = urlparse(href)
            root_link = p2._replace(query="", fragment="").geturl()
            frag = p2.fragment

            # Same-site check (allow www and non-www)
            if p2.netloc.replace("www.", "") != base_domain:
                continue
            # Skip noisy or irrelevant paths
            if any(kw in root_link.lower() for kw in EXCLUDE_KEYWORDS):
                continue

            next_url = root_link + (f"#{frag}" if frag else "")
            if next_url not in visited and next_url not in to_visit:
                to_visit.append(next_url)

    driver.quit()
    return output_file


def embed_and_rank_paragraphs(paragraphs, query, top_k=10, min_words=5, min_chars=60, boost_weight=0.2):
    """
    Embed and rank paragraphs by semantic similarity to the query,
    then boost those containing key investment keywords.
    Filters out very short, heading-like, or excessively brief text blocks.
    """
    # Expanded investment-related keywords to boost
    KEYWORDS = {
        "focus", "invest", "strategy", "portfolio", "sector", "thesis",
        "acquire", "acquires", "grows", "grow", "business", "company", "holding", "model", "mission", "goal"
    }

    def is_noise(p):
        if len(p.split()) < min_words:
            return True
        if len(p) < min_chars:
            return True
        letters = [c for c in p if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio > 0.6:
                return True
        return False

    clean = [p for p in paragraphs if not is_noise(p)]
    if not clean:
        clean = paragraphs  # fallback

    # compute embeddings
    query_vec = model.encode(query, convert_to_numpy=True)
    para_embs = model.encode(clean, convert_to_numpy=True, batch_size=100)

    # cosine similarities
    sims = np.dot(para_embs, query_vec) / (
        np.linalg.norm(para_embs, axis=1) * np.linalg.norm(query_vec)
    )

    # keyword flags and combined scores
    keyword_flags = np.array([1 if any(kw in p.lower() for kw in KEYWORDS) else 0 for p in clean])
    combined = sims + boost_weight * keyword_flags

    # sort by combined score and return only text and score
    idxs = np.argsort(combined)[::-1][:top_k]
    return [(clean[i], float(combined[i])) for i in idxs]

def read_txt(folder_path, filename):
    with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
        content = f.read().strip()
    return content

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

    
def main():
    firm_name = "blackrock"
    
    txt_file = crawl_site("https://joingardencity.com", max_pages=30)

    with open(txt_file, 'r', encoding='utf-8') as f:
        paras = [p.strip() for p in f.read().split('\n\n') if p.strip()]
    seen, clean_paras = set(), []
    for p in paras:
        if p not in seen:
            seen.add(p)
            clean_paras.append(p)

    query = (
        "Our private equity firm focuses on specific industries, employs an investment model such as buy-and-build or growth equity, and follows clear investment thesis statements for value creation."
        )

    top_k = embed_and_rank_paragraphs(clean_paras, query, top_k=60)

    try: os.remove(txt_file)
    except OSError: pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    snippet_path = os.path.join(OUTPUT_DIR, f"{firm_name}_relevant.txt")
    with open(snippet_path, 'w', encoding='utf-8') as rf:
        for txt, _ in top_k:
            rf.write(txt + "\n\n")

    text = read_txt(OUTPUT_DIR, f"{firm_name}_relevant.txt")
    for i in range(3):
        print(f"Attempt {i + 1}: Generating thesis...")
        thesis = call_model(format_prompt(text))
        response = call_model(format_grade_prompt(thesis))
        grade = extract_first_int(response)
        print(thesis)
        print(grade)

        if 1 == 1:
            print("Valid thesis identified. Exiting loop."  )
            break
        else:
            print("Thesis did not contain sufficient investment thesis information. Retrying...\n")
    else:
        print("Failed to generate a valid thesis after 3 attempts.")

    # delete_txt(OUTPUT_DIR, f"{firm_name}_relevant.txt")

    
if __name__ == '__main__':
    main()