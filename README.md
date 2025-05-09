# FaceCrime Backend V2

Backend for the FaceCrime facial recognition system. This service provides APIs for face matching, embedding generation, and adding new images to the database.

## Server Setup and Deployment

### Starting the Server

To start the server, use the provided `start_server.sh` script:

```bash
./start_server.sh
```

This script will:
1. Start the required Docker containers
2. Set up Tailscale funnel for port 8443
3. Verify that the service is running correctly
4. Display the public URL where the service is accessible

### Stopping the Server

To stop the server:

```bash
docker compose down
```

To view logs:

```bash
docker compose logs -f
```

## Frontend Development Guide

The backend provides three main endpoints that the frontend can interact with:

### 1. `/submission` Endpoint

**Purpose**: Process an image and find the most similar face in the database

**Request Format**:
```json
{
  "image": "base64_encoded_image_data"
}
```

**Response Format**:
```json
{
  "row_id": 123,
  "id": "ABC123",
  "prefix": "Mr",
  "firstname": "John",
  "middlename": "Robert",
  "lastname": "Doe",
  "suffix": "Jr",
  "gender": "Male",
  "dob": "1985-01-15",
  "location_name": "Central Prison",
  "location_type": "Correctional Facility",
  "streetaddress": "123 Prison Rd",
  "city": "Raleigh",
  "county": "Wake",
  "state": "NC",
  "zipcode": "27610",
  "latitude": 35.7796,
  "longitude": -78.6382,
  "offenderuri": "https://example.com/offender/123",
  "imageuri": "https://example.com/images/123.jpg",
  "absconder": false,
  "jurisdictionid": "NC123",
  "imagebase64": "base64_encoded_image_data",
  "matchPercent": 0.875
}
```

If no match is found, the response will be:
```json
{
  "results": []
}
```

### 2. `/add-image` Endpoint

**Purpose**: Add a new image and its metadata to the database

**Request Format**:
```json
{
  "row_id": 456,
  "id": "DEF456",
  "prefix": "Ms",
  "firstname": "Jane",
  "middlename": "Marie",
  "lastname": "Smith",
  "suffix": "",
  "gender": "Female",
  "dob": "1990-05-20",
  "location_name": "County Jail",
  "location_type": "Detention Center",
  "streetaddress": "456 Jail St",
  "city": "Charlotte",
  "county": "Mecklenburg",
  "state": "NC",
  "zipcode": "28202",
  "latitude": 35.2271,
  "longitude": -80.8431,
  "offenderuri": "https://example.com/offender/456",
  "imageuri": "https://example.com/images/456.jpg",
  "absconder": true,
  "jurisdictionid": "NC456",
  "imagebase64": "base64_encoded_image_data",
  "embedding": [0.1, 0.2, 0.3, ... ] // 512-dimensional vector
}
```

**Response Format**:
```json
{
  "id": "DEF456",
  "message": "Image + metadata added successfully"
}
```

### 3. `/embedding` Endpoint

**Purpose**: Generate a face embedding vector from an image

**Request Format**:
```json
{
  "image": "base64_encoded_image_data"
}
```

**Response Format**:
```json
{
  "embedding": [0.1, 0.2, 0.3, ... ] // 512-dimensional vector
}
```

## Integration Flow

A typical frontend integration flow would be:

1. **Face Matching**:
   - Capture or upload an image
   - Send the base64-encoded image to the `/submission` endpoint
   - Display the matching person's details and match percentage

2. **Adding New Faces**:
   - Capture or upload an image
   - Send the base64-encoded image to the `/embedding` endpoint to get its embedding vector
   - Collect all required metadata
   - Send the complete data to the `/add-image` endpoint

## Error Handling

All endpoints will return appropriate HTTP status codes:

- `200 OK`: Request successful
- `400 Bad Request`: Missing or invalid parameters
- `500 Internal Server Error`: Server-side error

Error responses will include a message explaining the issue:

```json
{
  "detail": "Missing 'image' field in request"
}
```