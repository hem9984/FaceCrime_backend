#!/usr/bin/env python3
# add_embeddings_pg.py
"""
Read a CSV with an 'imageBase64' column, POST each image to /embedding,
and write out a new CSV containing *only* the rows that succeeded,
with an extra 'embedding' column in pgvector format.
"""

import argparse
import csv
import json
import logging
import sys
from typing import Optional

import requests

logger = logging.getLogger(__name__)

csv.field_size_limit(sys.maxsize)

def fetch_embedding(b64: str, url: str, timeout: int) -> Optional[list]:
    """
    POST {"image": b64} to the embedding endpoint, return list[float] or None.
    """
    try:
        r = requests.post(url, json={"image": b64}, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        emb = j.get("embedding")
        if not isinstance(emb, list) or len(emb) == 0:
            return None
        return emb
    except Exception as e:
        logger.warning(f"Embedding request failed: {e}")
        return None

def main():
    p = argparse.ArgumentParser(
        description="Emit CSV of only those rows with a valid face embedding."
    )
    p.add_argument("-i","--input",  required=True,
                   help="Input CSV (must include imageBase64 column)")
    p.add_argument("-o","--output", required=True,
                   help="Output CSV (will include embedding column)")
    p.add_argument("-u","--embed-url",
                   default="http://localhost:8443/embedding",
                   help="Your backend’s /embedding URL")
    p.add_argument("-t","--timeout", type=int, default=30,
                   help="Request timeout in seconds")
    args = p.parse_args()

    # read input
    with open(args.input, newline="", encoding="utf-8") as inf:
        reader = csv.DictReader(inf)
        if "imageBase64" not in reader.fieldnames:
            logger.error("input CSV missing 'imageBase64' column")
            sys.exit(1)

        out_fields = list(reader.fieldnames) + ["embedding"]
        with open(args.output, "w", newline="", encoding="utf-8") as outf:
            writer = csv.DictWriter(outf, fieldnames=out_fields, 
                                    quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()

            total = 0
            success = 0

            for row in reader:
                total += 1
                b64 = row["imageBase64"].strip()
                if not b64:
                    continue

                emb = fetch_embedding(b64, args.embed_url, args.timeout)
                if emb is None:
                    continue

                # format as pgvector array "[v1,v2,...]" with no spaces
                row["embedding"] = json.dumps(emb, separators=(",",","))
                writer.writerow(row)
                success += 1

                if success % 100 == 0:
                    logger.info(f"✔ embedded {success}/{total}")

    logger.info(f"✨ Done — embedded {success} of {total} rows.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="[%H:%M:%S]"
    )
    main()

