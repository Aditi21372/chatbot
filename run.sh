#!/bin/bash

# Start the FastAPI backend
# Start the FastAPI backend
uvicorn app:app --reload &

# Access the backend at this link
echo "Backend is running at http://127.0.0.1:8000"

# Access the frontend at this link
echo "Frontend is running at http://127.0.0.1:8001/websocket_test.html"

# Start the HTTP server for the frontend
python3 -m http.server 8001

# Start the HTTP server for the frontend
python3 -m http.server 8001
