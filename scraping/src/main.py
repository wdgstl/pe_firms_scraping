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

    
        # 1) Read *all* lines (so blank lines are preserved)
        # 1) Read all lines, preserving blank ones
        # 1) Read raw lines (including blank ones and page‐break markers)
        with open(txt_file, 'r', encoding='utf-8') as f:
            file_lines = f.read().splitlines()

        # 2) Chunk on blank lines / headers / page breaks
        chunks = chunk_text(file_lines)

        # 3) Deduplicate the resulting chunks (preserving order)
        seen = set()
        clean_chunks = []
        for chunk in chunks:
            if chunk not in seen:
                seen.add(chunk)
                clean_chunks.append(chunk)

        query = "Industries: Healthcare, Software, Fintech, Retail, Agriculture, Biotech"

        # 4) Score the deduped chunks
        scored_chunks = embed_and_rank_paragraphs(clean_chunks, query, top_k=10)

        # 4) Write out
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        snippet_path = os.path.join(OUTPUT_DIR, f"{firm_name}_relevant.txt")
        with open(snippet_path, 'w', encoding='utf-8') as rf:
            for chunk, score in scored_chunks:
                rf.write(f"[{score}]{chunk}\n\n")

        model_queue.put((firm, snippet_path, txt_file))
        print(f"[{firm_name}] Relevant snippets written to {snippet_path}")

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

        firm, path, txt = item
        firm_name = firm['name']
        try:
            text = read_txt(OUTPUT_DIR, f"{firm_name}_relevant.txt")
            output = ""
            for i in range(3):
                print(f"[{firm_name}] Attempt {i+1}: Generating output...")
                draft = call_model(format_prompt(text))
                grade_resp = call_model(format_grade_prompt(draft))
                grade = extract_first_int(grade_resp)
                if grade == 1:
                    output = draft
                    print(f"[{firm_name}] Valid output found.")
                    break
                else:
                    print(f"[{firm_name}] Output insufficient, retrying...")

            # delete_txt(OUTPUT_DIR, f"{firm_name}_relevant.txt")

            industries = extract_industries(output)

            industries_thesis_map = {}

            for ind in industries:
                with open(txt, 'r', encoding='utf-8') as f:
                    file_lines = f.read().splitlines()

                chunks = chunk_text(file_lines)

                seen = set()
                clean_chunks = []
                for chunk in chunks:
                    if chunk not in seen:
                        seen.add(chunk)
                        clean_chunks.append(chunk)

                thesis_query = f"What is the investment thesis for {ind}?"

                # 4) Score the deduped chunks
                scored_chunks = embed_and_rank_paragraphs_thesis(clean_chunks, thesis_query, ind, top_k=10)

                # 4) Write out
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                snippet_path = os.path.join(OUTPUT_DIR, f"{firm_name}_{ind}_relevant.txt")
                with open(snippet_path, 'w', encoding='utf-8') as rf:
                    for chunk, score in scored_chunks:
                        rf.write(f"[{score}]{chunk}\n\n")
                print(f"[{firm_name}] Relevant snippets written to {snippet_path}")

                text = read_txt(OUTPUT_DIR, f"{firm_name}_{ind}_relevant.txt")
                thesis_raw = call_model(format_thesis_prompt(ind, text))

                thesis = extract_thesis(thesis_raw)

                industries_thesis_map[ind] = thesis


            db = SQLConnection(host, port, database, user, password)
            for ind in industries_thesis_map.keys():
                thesis = industries_thesis_map[ind]
                db.save_firm_to_db(
                    firm_name, firm['website'], ind, thesis,
                    firm.get('country', ''), str(firm.get('founded', '')),
                    firm.get('industry', ''), firm.get('linkedin_url', ''),
                    firm.get('locality', ''), firm.get('region', ''), firm.get('size', '')
                )
            db.close()
            print(f"[{firm_name}] Saved to database.")

        except Exception as e:
            print(f"[{firm_name}] Error in model processing: {e}")


def main(parallel: bool = False):    

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
        if parallel:                          # old behaviour
           with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            for firm in firms:
                executor.submit(process_firm, firm, model_queue)
        else:                                 # single-process scrape
            for firm in firms:
                process_firm(firm, model_queue)

        # Signal the worker to shut down
        model_queue.put(None)
        model_proc.join()


if __name__ == '__main__':
    main()
