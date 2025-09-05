## Sáttmál

Sáttmál (Old Norse for agreement, contract) is a lightweight, standalone document signing and analysis app built with Flask. It lets you upload PDFs, sign them directly in the browser, and optionally analyse their contents with AI to simplify what you’re agreeing to.

# Features

Upload and manage PDF

Draw or upload signatures directly in the browser

Store documents and signatures locally (JSON + filesystem) — no external DB needed

Install as a Progressive Web App (PWA) on mobile or desktop

Optional document “Explain Mode” using AI (Ollama or OpenAI) to simplify legal text

# Tech Stack

Python (Flask)

Bootstrap 5 (frontend styling)

Service Worker + Web Manifest (PWA support)

Local storage (filesystem + JSON file database)

Optional: Ollama / OpenAI for natural language analysis

Getting Started
1. Clone the repository
```bash
git clone https://github.com/YOURORG/sct-sáttmál.git
cd sct-sáttmál
```

2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the app
```bash
python app.py
```

By default, the app runs on http://localhost:5000

# PWA Support

Sáttmál can be installed as a Progressive Web App:

Open http://localhost:5000
 in Chrome or Edge.

Use “Install” or “Add to Home Screen.”

Launch it like a native app — fullscreen, offline-capable, background sync for signatures.

# Optional: AI Document Analysis

If you want document explanation:

Install Ollama locally and run ollama pull llama2.

Or set your OpenAI API key as an environment variable:

export OPENAI_API_KEY=your_key_here


Then, use /analyse_doc/<doc_id> to get simplified text for what you are signing.

# Open Code License

Sáttmál is released under our Open Code License:

The source code is free to use, modify, and extend.

Contributions are welcome via pull requests.

Any derivative work must credit the original project.

No warranty is provided; use at your own risk.

This license ensures openness, encourages collaboration, and maintains the integrity of the project.

# See LICENSE
 for the full text of the Open Code License.

# Contributing

Pull requests, ideas, and issues are always welcome.
Please read the CONTRIBUTING.md (coming soon) before submitting major changes.


# generate key once
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# copy the printed key

# set it in your shell or .env
export FERNET_KEY="PASTE_YOUR_KEY_HERE"

