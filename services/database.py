#!/usr/bin/env python3
import os
import logging
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

# DB settings from your .env
DB_HOST = os.environ.get("HARRISON_DB_HOST", "localhost")
DB_PORT = os.environ.get("HARRISON_DB_PORT", "5432")
DB_NAME = os.environ.get("HARRISON_DB_DB", "pomudatabase")
DB_USER = os.environ.get("HARRISON_DB_USER", "postgres")
DB_PASSWORD = os.environ.get("HARRISON_DB_PASSWORD", "")

_conn_str = (
    f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} "
    f"user={DB_USER} password={DB_PASSWORD}"
)

def get_connection():
    return psycopg2.connect(_conn_str)

def insert_image_and_metadata(
    row_id: int,
    id: str,
    prefix: str,
    firstname: str,
    middlename: str,
    lastname: str,
    suffix: str,
    gender: str,
    dob: str,
    location_name: str,
    location_type: str,
    streetaddress: str,
    city: str,
    county: str,
    state: str,
    zipcode: str,
    latitude: float,
    longitude: float,
    offenderuri: str,
    imageuri: str,
    absconder: bool,
    jurisdictionid: str,
    imagebase64: str,
    embedding: list[float],
):
    """
    Insert one row into facecrime_data.
    Assumes table has been created with columns matching these fields,
    embedding is vector(512) and indexed with HNSW using vector_cosine_ops.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO facecrime_data (
                      row_id, id, prefix, firstname, middlename, lastname, suffix,
                      gender, dob, location_name, location_type, streetaddress,
                      city, county, state, zipcode, latitude, longitude,
                      offenderuri, imageuri, absconder, jurisdictionid,
                      imagebase64, embedding
                    ) VALUES (
                      %s, %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s,
                      %s, %s, %s, %s, %s, %s,
                      %s, %s, %s, %s,
                      %s, %s::vector(512)
                    )
                    ON CONFLICT (row_id) DO NOTHING
                    """,
                    (
                        row_id, id, prefix, firstname, middlename, lastname, suffix,
                        gender, dob, location_name, location_type, streetaddress,
                        city, county, state, zipcode, latitude, longitude,
                        offenderuri, imageuri, absconder, jurisdictionid,
                        imagebase64, embedding
                    )
                )
        logger.debug(f"Inserted metadata for row_id={row_id}")
    except Exception as e:
        logger.error(f"Failed to insert row_id={row_id}: {e}")

def find_similar_image(embedding: list[float], limit: int = 1):
    """
    Retrieves the top `limit` matches by:
      1) Approximate HNSW pass: ORDER BY embedding <=> query LIMIT 50
      2) Exact re-ranking of those 50 by the same distance
    Returns a list of dicts with matchPercent in [0..1].
    """
    TOP_K = 50
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                sql = f"""
                SET vector_search_parameters = 'hnsw.ef=800';
                
                WITH candidates AS (
                  SELECT
                    row_id,
                    id,
                    prefix,
                    firstname,
                    middlename,
                    lastname,
                    suffix,
                    gender,
                    dob,
                    location_name,
                    location_type,
                    streetaddress,
                    city,
                    county,
                    state,
                    zipcode,
                    latitude,
                    longitude,
                    offenderuri,
                    imageuri,
                    absconder,
                    jurisdictionid,
                    imagebase64,
                    embedding <=> %s::vector(512) AS dist
                  FROM facecrime_data
                  ORDER BY embedding <=> %s::vector(512)
                  LIMIT {TOP_K}
                )
                SELECT
                  row_id,
                  id,
                  prefix,
                  firstname,
                  middlename,
                  lastname,
                  suffix,
                  gender,
                  dob,
                  location_name,
                  location_type,
                  streetaddress,
                  city,
                  county,
                  state,
                  zipcode,
                  latitude,
                  longitude,
                  offenderuri,
                  imageuri,
                  absconder,
                  jurisdictionid,
                  imagebase64,
                  (1 - dist) AS match_percent
                FROM candidates
                ORDER BY dist ASC
                LIMIT %s;
                """
                # pass the query vector three times: two for the WITH, one for final LIMIT
                cur.execute(sql, (embedding, embedding, limit))
                rows = cur.fetchall()

        results = []
        for row in rows:
            m = float(row["match_percent"])
            # clamp to [0..1]
            m = max(0.0, min(1.0, m))
            results.append({
                "row_id":         row["row_id"],
                "id":             row["id"],
                "prefix":         row["prefix"],
                "firstname":      row["firstname"],
                "middlename":     row["middlename"],
                "lastname":       row["lastname"],
                "suffix":         row["suffix"],
                "gender":         row["gender"],
                "dob":            row["dob"],
                "location_name":  row["location_name"],
                "location_type":  row["location_type"],
                "streetaddress":  row["streetaddress"],
                "city":           row["city"],
                "county":         row["county"],
                "state":          row["state"],
                "zipcode":        row["zipcode"],
                "latitude":       row["latitude"],
                "longitude":      row["longitude"],
                "offenderuri":    row["offenderuri"],
                "imageuri":       row["imageuri"],
                "absconder":      row["absconder"],
                "jurisdictionid": row["jurisdictionid"],
                "imagebase64":    row["imagebase64"],
                "matchPercent":   m,
            })
        return results

    except Exception as e:
        logger.error(f"Similarity query failed: {e}")
        return []
