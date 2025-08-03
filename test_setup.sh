#!/bin/bash

echo "Testing Google Cloud setup..."

# Set credentials
export GOOGLE_APPLICATION_CREDENTIALS="./credentials/key.json"

# Test credentials
python3 -c "
try:
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()
    print('✅ Google Cloud Vision client works')
except Exception as e:
    print('❌ Google Cloud setup failed:', e)
    exit(1)
"
