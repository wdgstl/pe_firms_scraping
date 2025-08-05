import os
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager, Process
from llama import *
from scrape import *
from sql import *

CSV_PATH = 'pefirms.csv'
OUTPUT_DIR = 'output'

# Load environment and database config
load_dotenv()
host = os.environ['PG_HOST_local']
port = os.environ['PG_PORT']
database = os.environ['PG_DATABASE']
user = os.environ['PG_USER']
password = os.environ['PG_PASSWORD']


def get_firms(path):
    df = pd.read_csv(path)
    return df[(df['country'].str.lower() == 'united states') & (df['website'].notna())].to_dict(orient='records')


def process_firm(firm, model_queue):
    firm_id = str(firm['id'])
    firm_name = firm['name']
    firm_website = 'https://' + firm['website'].strip()

    print(f"[{firm_id}] Visiting {firm_website}")
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
        "Our private equity firm focuses on specific industries, employs an investment model such as buy-and-build or growth equity, and follows clear investment thesis statements for value creation."
        )
        top_k = embed_and_rank_paragraphs(clean_paras, query, top_k=60)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        snippet_path = os.path.join(OUTPUT_DIR, f"{firm_name}_relevant.txt")
        with open(snippet_path, 'w', encoding='utf-8') as rf:
            for txt, _ in top_k:
                rf.write(txt + "\n\n")

        model_queue.put((firm, snippet_path))
        print(f"[{firm_name}] Enqueued for model processing.")

    except Exception as e:
        print(f"[{firm_id}] Error in scraping: {e}")


def model_worker(model_queue):
    print("[Model Worker] Started and waiting for queue items.")
    while True:
        item = model_queue.get()
        # Shutdown signal
        if item is None:
            print("[Model Worker] Shutdown signal received. Exiting.")
            break

        firm, path = item
        firm_id = str(firm['id'])
        firm_name = firm['name']
        try:
            text = read_txt(OUTPUT_DIR, f"{firm_name}_relevant.txt")
            thesis = ""
            for i in range(3):
                print(f"[{firm_name}] Attempt {i+1}: Generating thesis...")
                draft = call_model(format_prompt(text))
                grade_resp = call_model(format_grade_prompt(draft))
                grade = extract_first_int(grade_resp)
                if grade == 1:
                    thesis = draft
                    print(f"[{firm_name}] Valid thesis found.")
                    break
                else:
                    print(f"[{firm_name}] Thesis insufficient, retrying...")

            delete_txt(OUTPUT_DIR, f"{firm_name}_relevant.txt")

            db = SQLConnection(host, port, database, user, password)
            db.save_firm_to_db(
                firm_id, firm_name, firm['website'], thesis,
                firm.get('country', ''), str(firm.get('founded', '')),
                firm.get('industry', ''), firm.get('linkedin_url', ''),
                firm.get('locality', ''), firm.get('region', ''), firm.get('size', '')
            )
            db.close()
            print(f"[{firm_name}] Saved to database.")

        except Exception as e:
            print(f"[{firm_name}] Error in model processing: {e}")


def main():
    # Initialize database
    db = SQLConnection(host, port, database, user, password)
    db.drop_table()
    db.create_table()
    db.close()

    firms = get_firms(CSV_PATH)

    # Use a Manager queue for inter-process communication
    with Manager() as manager:
        model_queue = manager.Queue(maxsize=100)

        # Start model worker process
        model_proc = Process(target=model_worker, args=(model_queue,))
        model_proc.start()

        # Scrape firms in parallel
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            for firm in firms:
                executor.submit(process_firm, firm, model_queue)

        # Signal the worker to shut down
        model_queue.put(None)
        model_proc.join()


if __name__ == '__main__':
    main()
