from llama import * 
from scrape import *
from sql import * 
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
import os
from dotenv import load_dotenv

CSV_PATH = 'pefirms.csv'

load_dotenv()
host = os.environ["PG_HOST_local"]
port = os.environ["PG_PORT"]
database = os.environ["PG_DATABASE"]
user = os.environ["PG_USER"]
password = os.environ["PG_PASSWORD"]


def get_firms(path):
    df = pd.read_csv(path)
    return df[
        (df['country'].str.lower() == 'united states') &
        (df['website'].notna())
    ].to_dict(orient='records')

def process_firm(firm):
    firm_id       = str(firm['id'])
    firm_name     = firm['name']
    firm_website  = 'https://' + firm['website'].strip()
    firm_country  = firm.get('country', '')
    firm_founded  = str(firm.get('founded', ''))
    firm_industry = firm.get('industry', '')
    firm_linkedin = firm.get('linkedin_url', '')
    firm_locality = firm.get('locality', '')
    firm_region   = firm.get('region', '')
    firm_size     = firm.get('size', '')

    print(f"[{firm_id}] Starting")

    thesis = ""
    try:
        txt_file = crawl_site(firm_website, max_pages=30)

        with open(txt_file, 'r', encoding='utf-8') as f:
            paras = [p.strip() for p in f.read().split('\n\n') if p.strip()]
        seen, clean_paras = set(), []
        for p in paras:
            if p not in seen:
                seen.add(p)
                clean_paras.append(p)

        query = (
            f"{firm_name} private equity industry focus areas, "
            "investment model, and corresponding investment thesis statements."
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
        thesis = call_mixtral(text)
        delete_txt(OUTPUT_DIR, f"{firm_name}_relevant.txt")

    except Exception as e:
        print(f"[{firm_id}] Error in processing: {e}")

    try:
        db = SQLConnection(host, port, database, user, password)
        db.save_firm_to_db(
            firm_id, firm_name, firm_website, thesis,
            firm_country, firm_founded, firm_industry,
            firm_linkedin, firm_locality, firm_region, firm_size
        )
    except Exception as e:
        print(f"[{firm_id}] DB error: {e}")
    finally:
        db.close()

    print(f"[{firm_id}] Done")

num_cores = os.cpu_count() or 1
PARALLEL  = False   # ← flip this to False to run sequentially

def main():
    firms = get_firms(CSV_PATH)
    db = SQLConnection(host, port, database, user, password)
    db.drop_table()
    db.create_table()
    db.close()

    if PARALLEL:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            executor.map(process_firm, firms)
    else:
        for firm in firms:
            process_firm(firm)

if __name__ == '__main__':
    main()
