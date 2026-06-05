# deriv_interview
 This is the coding problem i solved for deriv.

# First, install the requirements.
pip install -r requirements.txt

# Start the app using this command
uvicorn main:app --reload

# Open another terminal and enter this command to start using the app
curl -X POST http://127.0.0.1:8000/index

# If the document folder exits, and all the requirements are satisfised, the success output will be like this
{"status":"success","documents_indexed":3,"chunks_created":3}

# To ask questions, use the following command
curl -X POST http://127.0.0.1:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "YOUR QUESTION"}'
