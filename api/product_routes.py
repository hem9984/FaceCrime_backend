# api/product_routes.py

import logging
from fastapi import APIRouter, HTTPException, Request
from services.face_rec import extract_face_embedding
from services.database import find_similar_image, insert_image_and_metadata
from typing import List, Optional

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/submission")
async def submission(request: Request):
    """
    Process incoming image, get its embedding, 
    find the most similar face in the DB, 
    and return all attributes in the JSON format the frontend expects.
    """
    data = await request.json()
    if 'image' not in data:
        raise HTTPException(400, "Missing 'image' field in request")

    emb = extract_face_embedding(data['image'])
    if not emb:
        raise HTTPException(400, "Failed to process image")

    sims = find_similar_image(emb, limit=1)
    if not sims:
        return {"results": []}

    # Return all fields from the database to the frontend
    top = sims[0]
    top["matchPercent"] = round(top["matchPercent"], 3)
    return top

@router.post("/add-image")
async def add_image(request: Request):
    """
    Insert a new image + metadata into the DB.
    """
    data = await request.json()
    
    # Required fields based on database schema
    required = [
        "row_id", "id", "prefix", "firstname", "middlename", "lastname", 
        "suffix", "gender", "dob", "location_name", "location_type", 
        "streetaddress", "city", "county", "state", "zipcode", 
        "latitude", "longitude", "offenderuri", "imageuri", 
        "absconder", "jurisdictionid", "imagebase64", "embedding"
    ]
    
    for f in required:
        if f not in data:
            raise HTTPException(400, f"Missing '{f}' in request")
    
    # Pass all parameters to match database.py structure
    insert_image_and_metadata(
        row_id=data["row_id"],
        id=data["id"],
        prefix=data["prefix"],
        firstname=data["firstname"],
        middlename=data["middlename"],
        lastname=data["lastname"],
        suffix=data["suffix"],
        gender=data["gender"],
        dob=data["dob"],
        location_name=data["location_name"],
        location_type=data["location_type"],
        streetaddress=data["streetaddress"],
        city=data["city"],
        county=data["county"],
        state=data["state"],
        zipcode=data["zipcode"],
        latitude=float(data["latitude"]),
        longitude=float(data["longitude"]),
        offenderuri=data["offenderuri"],
        imageuri=data["imageuri"],
        absconder=bool(data["absconder"]),
        jurisdictionid=data["jurisdictionid"],
        imagebase64=data["imagebase64"],
        embedding=data["embedding"]
    )
    
    return {"id": data["id"], "message": "Image + metadata added successfully"}

@router.post("/embedding")
async def embedding(request: Request):
    """
    Decode a base64 image and return its face embedding vector.
    """
    data = await request.json()
    if 'image' not in data:
        raise HTTPException(400, "Missing 'image' in request")
    emb = extract_face_embedding(data['image'])
    if emb is None or len(emb)==0:
        raise HTTPException(400, "No face embedding could be extracted")
    # return list of floats
    return {"embedding": emb}

