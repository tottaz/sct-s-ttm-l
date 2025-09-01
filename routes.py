from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    send_from_directory,
    url_for,
    flash,
    send_file,
    jsonify
)
import os
import uuid
import base64
import json
from datetime import datetime
from werkzeug.utils import secure_filename

from openai import OpenAI
import ollama

signature_bp = Blueprint("signature", __name__,
                         template_folder="templates",
                         static_folder="static")

CONFIG_FILE = os.path.join(os.getcwd(), "config.json")

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

USE_OPENAI = config.get("use_openai", True)
OPENAI_API_KEY = config.get("openai_api_key")
OLLAMA_BASE_URL = config.get("ollama_base_url", "http://localhost:11434")


UPLOADS_DIR = os.path.join(os.getcwd(), "data", "uploads")
SIGNATURES_DIR = os.path.join(os.getcwd(), "data", "signatures")

# Make sure directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(SIGNATURES_DIR, exist_ok=True)


def metadata_path_for(file_id: str) -> str:
    return os.path.join(UPLOADS_DIR, f"{file_id}.json")


def write_metadata(meta: dict):
    """Write metadata for a single file (meta must contain 'id')."""
    path = metadata_path_for(meta["id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def read_metadata(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _create_meta_for_existing_file(filename: str) -> dict:
    """Create metadata for a raw file found in uploads (migration helper)."""
    file_id = str(uuid.uuid4())
    stored_filename = filename
    file_path = os.path.join(UPLOADS_DIR, stored_filename)
    meta = {
        "id": file_id,
        "original_filename": filename,
        "stored_filename": stored_filename,
        "filename": filename,
        "file_path": file_path,
        "type": "pdf" if filename.lower().endswith(".pdf") else "signature",
        "status": "uploaded",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    # save metadata file next to it
    write_metadata(meta)
    return meta


def load_docs() -> list:
    """
    Return a list of metadata dicts for all documents.
    Priority: read per-file JSON metadata files. If no metadata exists
    but files are present, create metadata for them (migration).
    Result is sorted by timestamp desc (newest first).
    """
    docs = []
    # find any metadata JSON files in uploads dir
    for name in os.listdir(UPLOADS_DIR):
        if name.endswith(".json"):
            try:
                meta = read_metadata(os.path.join(UPLOADS_DIR, name))
                docs.append(meta)
            except Exception:
                # skip malformed metadata files
                continue

    # If we found no metadata but there are files, create metadata for each
    if not docs:
        for name in os.listdir(UPLOADS_DIR):
            if name.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
                docs.append(_create_meta_for_existing_file(name))

    # normalize & sort (if timestamp missing, put at end)
    def _ts(d):
        t = d.get("timestamp")
        if not t:
            return ""
        return t
    docs.sort(key=_ts, reverse=True)
    return docs


def save_metadata_for_doc_id(doc_id: str, updates: dict):
    """Load metadata for doc_id, update with `updates` dict and save."""
    path = metadata_path_for(doc_id)
    if not os.path.exists(path):
        raise FileNotFoundError("Metadata not found for id: " + doc_id)
    meta = read_metadata(path)
    meta.update(updates)
    write_metadata(meta)
    return meta


def analyze_pdfdoc(body: str) -> str:
    system_prompt = (
        "You are an AI assistant. "
        "Analyze the following PDF document content. "
        "Summarize the key points, identify important information, "
        "and highlight any notable sections or recommendations. "
        "Provide the analysis in a clear and concise manner."
    )

    if USE_OPENAI:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": body}
            ]
        )
        return response.choices[0].message.content.strip()
    else:
        # Ollama
        try:
            ollama.list()
        except Exception:
            raise Exception("Ollama server is not running. Start it with: ollama serve")
        try:
            response = ollama.chat(
                model="llama3.2:latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": body}
                ]
            )
            return response["message"]["content"].strip()
        except Exception as e:
            raise Exception(f"Ollama chat error: {e}")


# Default route → dashboard
@signature_bp.route("/")
def index():
    docs = load_docs()
    return render_template("docdashboard.html", docs=docs)


# View PDF
@signature_bp.route("/docupload", methods=["GET", "POST"])
def docupload():
    if request.method == "POST":
        pdf_file = request.files.get("pdf_file")

        file_id = str(uuid.uuid4())

        if pdf_file and pdf_file.filename:
            original_filename = secure_filename(pdf_file.filename)
            # store with file_id prefix to avoid collisions
            stored_filename = f"{file_id}_{original_filename}"
            file_path = os.path.join(UPLOADS_DIR, stored_filename)
            pdf_file.save(file_path)

            meta = {
                "id": file_id,
                "original_filename": original_filename,
                "stored_filename": stored_filename,
                "filename": original_filename,
                "file_path": file_path,
                "type": "pdf",
                "status": "uploaded",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            write_metadata(meta)

            flash("PDF uploaded successfully.", "success")
            return redirect(url_for("signature.docupload"))

        else:
            flash("Please provide a PDF file URL.", "danger")
            return redirect(url_for("signature.docupload"))

    # GET -> show current docs
    docs = load_docs()
    return render_template("docdashboard.html", docs=docs)


# Serve uploaded files
@signature_bp.route('/docuploaded_file/<path:filename>')
def docuploaded_file(filename):
    uploads_dir = os.path.join(os.getcwd(), "data", "uploads")
    return send_from_directory(uploads_dir, filename)


# Sign document (draw signature)
@signature_bp.route("/docsign/<doc_id>")
def docsign(doc_id):
    meta_path = os.path.join(UPLOADS_DIR, f"{doc_id}.json")
    if not os.path.exists(meta_path):
        flash("Document metadata not found.", "danger")
        return redirect(url_for("signature.index"))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    file_path = meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        flash("Document file not found on disk.", "danger")
        return redirect(url_for("signature.index"))

    # Provide a URL for the iframe to display the PDF
    meta["view_url"] = url_for("signature.serve_doc", doc_id=doc_id)

    return render_template("docsignature.html", document=meta)


# View PDF
@signature_bp.route("/docview/<doc_id>")
def docview(doc_id):
    meta_path = os.path.join(UPLOADS_DIR, f"{doc_id}.json")
    if not os.path.exists(meta_path):
        flash("Document metadata not found.", "danger")
        return redirect(url_for("signature.index"))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if meta.get("type") == "pdf":
        meta["view_url"] = url_for("signature.serve_doc", doc_id=doc_id)
    elif meta.get("type") == "google_doc":
        meta["view_url"] = meta.get("url")
    else:
        meta["view_url"] = None

    return render_template("docview.html", doc=meta)


@signature_bp.route("/docdelete/<doc_id>", methods=["POST"])
def docdelete(doc_id):
    meta_path = os.path.join(UPLOADS_DIR, f"{doc_id}.json")
    if not os.path.exists(meta_path):
        flash("Document metadata not found.", "danger")
        return redirect(url_for("signature.index"))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    file_path = meta.get("file_path")
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    # Delete the metadata JSON
    os.remove(meta_path)
    flash(f"Document '{meta.get('filename', doc_id)}' deleted successfully.", "success")
    return redirect(url_for("signature.index"))


@signature_bp.route("/download/<doc_id>")
def download_doc(doc_id):
    meta_path = os.path.join(UPLOADS_DIR, f"{doc_id}.json")
    if not os.path.exists(meta_path):
        flash("Document metadata not found.", "danger")
        return redirect(url_for("signature.index"))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    file_path = meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        flash("Document file not found on disk.", "danger")
        return redirect(url_for("signature.index"))

    return send_file(file_path, as_attachment=True, download_name=meta.get("filename", doc_id))


@signature_bp.route('/save_signed_pdf',
                    methods=['POST'])
def save_signed_pdf():
    import base64
    pdf_data = request.json.get('pdf_base64')
    original_filename = request.json.get('original_filename', 'document.pdf')

    if not pdf_data:
        return {"error": "No PDF data received"}, 400

    # Decode base64 PDF
    pdf_bytes = base64.b64decode(pdf_data.split(",")[1])

    # Create a unique filename
    signed_filename = f"signed_{uuid.uuid4()}_{original_filename}"
    file_path = os.path.join("data/uploads", signed_filename)

    # Save to disk
    with open(file_path, "wb") as f:
        f.write(pdf_bytes)

    return {"success": True, "filename": signed_filename}


# Save drawn signature
@signature_bp.route("/save_signature", methods=["POST"])
def save_signature():
    data = request.json
    sig_data = data.get("signature")
    doc_id = data.get("doc_id")

    if not sig_data or not doc_id:
        return {"success": False, "error": "Missing signature or document ID"}, 400

    sig_id = f"{uuid.uuid4()}.png"
    sig_path = os.path.join(SIGNATURES_DIR, sig_id)
    os.makedirs(SIGNATURES_DIR, exist_ok=True)

    with open(sig_path, "wb") as f:
        f.write(base64.b64decode(sig_data.split(",")[1]))

    # Update the document JSON to store signature path
    meta_path = os.path.join(UPLOADS_DIR, f"{doc_id}.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["signature"] = sig_path
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return {"success": True, "signature_url": url_for("signature.serve_signature", sig_id=sig_id)}


# Analyse document (AI stub)
@signature_bp.route("/analyze_doc/<doc_id>")
def analyze_doc(doc_id):
    meta_path = os.path.join(UPLOADS_DIR, f"{doc_id}.json")
    if not os.path.exists(meta_path):
        return {"success": False, "analysis": "Document metadata not found."}

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    file_path = meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return {"success": False, "analysis": "Document file not found."}

    # Extract text from PDF (you can use PyPDF2 / pdfplumber)
    import pdfplumber
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    analysis = analyze_pdfdoc(text)
    return {"success": True, "analysis": analysis}



@signature_bp.route("/list_signatures")
def list_signatures():
    files = os.listdir(SIGNATURES_DIR)
    # optionally filter only PNGs
    files = [f for f in files if f.endswith(".png")]
    return jsonify([{"id": f} for f in files])


@signature_bp.route("/serve_signature/<sig_id>")
def serve_signature(sig_id):
    sig_path = os.path.join(SIGNATURES_DIR, sig_id)
    if not os.path.exists(sig_path):
        flash("Signature not found.", "danger")
        return redirect(url_for("signature.index"))
    return send_file(sig_path, mimetype="image/png")


@signature_bp.route("/serve/<doc_id>")
def serve_doc(doc_id):
    meta_path = os.path.join(UPLOADS_DIR, f"{doc_id}.json")
    if not os.path.exists(meta_path):
        flash("Document metadata not found.", "danger")
        return redirect(url_for("signature.index"))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    file_path = meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        flash("Document file not found on disk.", "danger")
        return redirect(url_for("signature.index"))

    # Serve PDF inline
    return send_file(file_path, mimetype='application/pdf')
