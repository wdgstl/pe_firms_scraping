# main_people.py
import os
import re
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
from people_scrape import LeadershipCrawler  # your existing crawler
from sql import SQLConnection          # <-- change to your actual module name
import pandas as pd
from urllib.parse import urlparse
import requests
import csv


UNREACHABLE_CSV='unreachable_people_links.csv'

# =======================
# Faith keyword detection
# =======================
FAITH_KEYWORDS = [
    r"faith", r"faith[- ]based",
    r"christian", r"christianity",
    r"catholic", r"protestant", r"evangelical",
    r"church", r"ministry", r"pastor",
    r"bible", r"scripture", r"gospel",
    r"god", r"lord", r"passage", r"psalm", r"proverb"
]

FAITH_REGEX = re.compile(r"(" + "|".join(FAITH_KEYWORDS) + r")", re.IGNORECASE)

SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+')

def compute_faith_score(text: str):
    if not text:
        return 0, ""

    # Find all sentences
    sentences = SENTENCE_SPLIT_REGEX.split(text)

    # Check for keyword matches and collect matching sentences
    matches = [s.strip() for s in sentences if FAITH_REGEX.search(s)]
    
    if matches:
        evidence = " ".join(matches)
        return 1, evidence
    else:
        return 0, ""


# =======================
# Title extraction (highest wins)
# =======================
TITLE_SYNONYMS: Dict[str, List[str]] = {
    # Board / C-suite
    "executive chair": [r"\bexecutive chair(?:man|woman)?\b"],
    "chair":           [r"\bchair(?:man|woman)?\b", r"\bboard chair\b"],
    "ceo":             [r"\bchief executive officer\b", r"\bceo\b"],
    "president":       [r"\bpresident\b"],
    "coo":             [r"\bchief operating officer\b", r"\bcoo\b"],
    "cfo":             [r"\bchief financial officer\b", r"\bcfo\b"],
    "cio":             [r"\bchief information officer\b", r"\bcio\b"],
    "cto":             [r"\bchief technology officer\b", r"\bcto\b"],
    "chief":           [r"\bchief [a-z][a-z /&\-]+\b"],

    # Partners / heads
    "managing partner": [r"\bmanaging partner\b"],
    "partner":          [r"\bgeneral partner\b", r"\boperating partner\b", r"\bpartner\b"],
    "head":             [r"\bglobal head\b", r"\bhead of [a-z][a-z /&\-]+\b", r"\bhead\b"],

    # MD / Director
    "managing director":[r"\bmanaging director\b", r"\bmd\b(?![a-z])"],
    "senior director":  [r"\bsenior director\b", r"\bsr\.?\s*director\b"],
    "director":         [r"\bdirector\b"],

    # VP tier
    "executive vice president": [r"\bexecutive vice president\b", r"\bevp\b"],
    "senior vice president":    [r"\bsenior vice president\b", r"\bsvp\b"],
    "vice president":           [r"\bvice president\b", r"\bvp\b", r"\bavp\b",
                                 r"\bass(?:t|istant)\s+vice president\b"],

    # Manager / ICs
    "senior manager":   [r"\bsenior manager\b", r"\bsr\.?\s*manager\b"],
    "manager":          [r"\bmanager\b", r"\bportfolio manager\b", r"\bproduct manager\b"],
    "principal":        [r"\bprincipal\b"],
    "senior associate": [r"\bsenior associate\b", r"\bsr\.?\s*associate\b"],
    "associate":        [r"\bassociate\b"],
    "senior analyst":   [r"\bsenior analyst\b"],
    "analyst":          [r"\banalyst\b"],
    "intern":           [r"\bintern\b"],

    # Founders
    "founder":          [r"\bfounder\b", r"\bco-?founder\b"],
}

# lower number = higher seniority
SENIORITY_RANK_HIGH = {
    "executive chair": 0,
    "chair": 1,
    "ceo": 2,
    "president": 3,
    "coo": 4, "cfo": 4, "cio": 5, "cto": 5, "chief": 6,
    "managing partner": 7,
    "partner": 8,
    "head": 9,
    "managing director": 10,
    "senior director": 11,
    "director": 12,
    "executive vice president": 13,
    "senior vice president": 14,
    "vice president": 15,
    "senior manager": 16,
    "manager": 17,
    "principal": 18,
    "senior associate": 19,
    "associate": 20,
    "senior analyst": 21,
    "analyst": 22,
    "intern": 23,
    "founder": 6,  # tweak if you want founder to outrank C-suite
}

TITLE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    (canon, re.compile(r"(?:" + "|".join(syns) + r")", re.IGNORECASE))
    for canon, syns in TITLE_SYNONYMS.items()
]

def extract_best_position_from_text(text: str) -> Optional[str]:
    """Return the highest-seniority canonical title mentioned in text."""
    if not text:
        return None
    found: List[str] = []
    for canon, pat in TITLE_PATTERNS:
        if pat.search(text):
            found.append(canon)
    if not found:
        return None
    return min(found, key=lambda c: SENIORITY_RANK_HIGH.get(c, 9999))

def choose_best_position(heading_pos: str, bio_text: str) -> str:
    """Pick highest rank across heading-adjacent title and any titles found in the bio."""
    heading_pos = (heading_pos or "").strip()
    bio_pos = extract_best_position_from_text(bio_text or "")
    if not heading_pos and not bio_pos:
        return ""
    if heading_pos and not bio_pos:
        return heading_pos
    if bio_pos and not heading_pos:
        return bio_pos
    # both exist -> choose higher (lower rank number)
    h = SENIORITY_RANK_HIGH.get(heading_pos.lower(), 9999)
    b = SENIORITY_RANK_HIGH.get(bio_pos.lower(), 9999)
    return bio_pos if b < h else heading_pos


def normalize_url(url: str) -> str:
    """Ensure the URL has https:// and strip trailing slashes."""
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url
    return url.rstrip("/")



def is_reachable(test_url: str, csv_path: str = UNREACHABLE_CSV) -> bool:
    """
    Checks if test_url is reachable quickly (HEAD -> tiny GET fallback).
    If unreachable, logs (url, reason) to csv_path.
    """
    s = requests.Session()
    timeout = (2, 4)  # connect/read timeout

    try:
        # Try HEAD request first
        r = s.head(test_url, timeout=timeout, allow_redirects=True)
        if r.status_code in (405, 400) or ("HEAD" not in r.headers.get("Allow", "") and r.status_code >= 400):
            # Fallback: tiny GET
            g = s.get(test_url, headers={"Range": "bytes=0-0"}, timeout=timeout, allow_redirects=True, stream=True)
            g.close()
            if 200 <= g.status_code < 400:
                return True
            _write_unreachable(csv_path, test_url, f"GET {g.status_code}")
            return False

        if 200 <= r.status_code < 400:
            return True

        _write_unreachable(csv_path, test_url, f"HEAD {r.status_code}")
        return False

    except requests.exceptions.SSLError:
        # Retry with http:// if SSL fails
        try:
            http_url = "http://" + test_url.split("://", 1)[-1]
            r2 = s.head(http_url, timeout=timeout, allow_redirects=True)
            if r2.status_code in (405, 400) or ("HEAD" not in r2.headers.get("Allow", "") and r2.status_code >= 400):
                g2 = s.get(http_url, headers={"Range": "bytes=0-0"}, timeout=timeout, allow_redirects=True, stream=True)
                g2.close()
                if 200 <= g2.status_code < 400:
                    return True
                _write_unreachable(csv_path, test_url, f"HTTP GET {g2.status_code}")
                return False
            if 200 <= r2.status_code < 400:
                return True
            _write_unreachable(csv_path, test_url, f"HTTP HEAD {r2.status_code}")
            return False
        except Exception as e2:
            _write_unreachable(csv_path, test_url, f"http-retry-failed: {type(e2).__name__}")
            return False

    except (requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError) as e:
        _write_unreachable(csv_path, test_url, type(e).__name__)
        return False

    except requests.exceptions.RequestException as e:
        _write_unreachable(csv_path, test_url, f"req-exc:{type(e).__name__}")
        return False


def _write_unreachable(csv_path: str, url: str, reason: str) -> None:
    """Append (url, reason) to CSV; create with header if missing."""
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["url", "reason"])
        w.writerow([url, reason])
    

# =======================
# Pipeline to your DB
# =======================
def run_people_pipeline_to_db(firm: dict, db_conn) -> None:
    """
    Crawl a firm's leadership pages, compute faith score & best position,
    and insert/update people in your Postgres via SQLConnection.

    """

    raw_site = firm.get('website', '')
    firm_website = normalize_url(raw_site)

   
    unreachable_set = set()
    if os.path.exists(UNREACHABLE_CSV):
        with open(UNREACHABLE_CSV, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                val = row[0].strip()
                if val.lower() != "url" and val:
                    unreachable_set.add(val)


    if firm_website in unreachable_set:
        print(f"[skip] Already in unreachable.csv: {firm_website}")
        return

    if not is_reachable(firm_website):
        print(f'Skipping {firm_website}... not reachable')
        return

    crawler = LeadershipCrawler(firm_website, max_pages=200, max_depth=3, verbose=True)
    bios = crawler.crawl()
    


    for rec in bios:
        name = (rec.get("name") or "").strip()
        if not name:
            continue

        bio = rec.get("bio", "") or ""
        heading_pos = rec.get("position") or ""
        best_pos = choose_best_position(heading_pos, bio)
        faith, evidence = compute_faith_score(bio)
        firm_name = firm.get('name', '')
        region =  firm.get('region', '')

        db_conn.save_person_to_db(
            name=name,
            firm= firm_name,
            region = region,
            position= best_pos,
            faith= faith,
            evidence=evidence
        )

def get_firms(path):
    df = pd.read_csv(path)
    return df[(df['country'].str.lower() == 'united states') & (df['website'].notna())].to_dict(orient='records')


# =======================
# Configure + Run
# =======================
if __name__ == "__main__":
    # 1) Target firm URL
    CSV_PATH = 'pefirms.csv'

    # 2) Load DB creds from .env and use YOUR SQLConnection
    load_dotenv()
    host = os.environ["PG_HOST_local"]
    port = os.environ["PG_PORT"]          # keep as string if your SQLConnection expects it
    database = os.environ["PG_DATABASE"]
    user = os.environ["PG_USER"]
    password = os.environ["PG_PASSWORD"]

    db = SQLConnection(host, port, database, user, password)

    # Ensure your SQLConnection exposes these methods exactly as you showed:
    # - create_people_table()
    # - save_person_to_db(name, firm, position, faith, evidence)

    db.create_people_table()

    firms = get_firms(CSV_PATH)
    for firm in firms:
        
        run_people_pipeline_to_db(firm, db)
      

