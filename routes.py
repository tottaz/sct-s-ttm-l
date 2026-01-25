from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    send_from_directory,
    url_for,
    flash,
    send_file,
    jsonify,
    abort
)
import os
import io
import sys
import uuid
import base64
import json
import shutil
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime
from werkzeug.utils import secure_filename
from cryptography.fernet import Fernet

from openai import OpenAI
import ollama
import docx
import markdown
import pdfplumber

signature_bp = Blueprint("signature", __name__,
                         template_folder="templates",
                         static_folder="static")

def get_data_dir():
    """Get a writable directory for application data."""
    if getattr(sys, 'frozen', False):
        # Running in a bundle (e.g., PyInstaller)
        if sys.platform == 'darwin':
            # macOS: ~/Library/Application Support/Sattmal
            base_dir = os.path.expanduser('~/Library/Application Support/Sattmal')
        else:
            # Fallback for other platforms if needed
            base_dir = os.path.join(os.path.expanduser('~'), '.sattmal')
    else:
        # Running in development
        base_dir = os.getcwd()
    
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return base_dir

DATA_BASE_DIR = get_data_dir()
CONFIG_FILE = os.path.join(DATA_BASE_DIR, "config.json")

# If config doesn't exist in the data dir, try to copy it from the bundle/source
if not os.path.exists(CONFIG_FILE):
    base_path = getattr(sys, '_MEIPASS', os.getcwd())
    src_config = os.path.join(base_path, "config.json")
    # Also check for config.example.json as a fallback
    src_example = os.path.join(base_path, "config.example.json")
    
    if os.path.exists(src_config):
        shutil.copy(src_config, CONFIG_FILE)
    elif os.path.exists(src_example):
        shutil.copy(src_example, CONFIG_FILE)
    else:
        # Create a default empty config if source is missing too
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "use_openai": False,
                "openai_api_key": "",
                "ollama_base_url": "http://localhost:11434",
                "fernet_key": Fernet.generate_key().decode()
            }, f, indent=2)

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

USE_OPENAI = config.get("use_openai", False)
OPENAI_API_KEY = config.get("openai_api_key")
OLLAMA_BASE_URL = config.get("ollama_base_url", "http://localhost:11434")
FERNET_KEY = config.get("fernet_key")

def is_valid_fernet_key(key):
    if not key or not isinstance(key, str):
        return False
    try:
        Fernet(key)
        return True
    except Exception:
        return False

if not is_valid_fernet_key(FERNET_KEY):
    # If missing or invalid (e.g. placeholder), generate a new one
    FERNET_KEY = Fernet.generate_key().decode()
    config["fernet_key"] = FERNET_KEY
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

# Initialize cipher
cipher = Fernet(FERNET_KEY)

UPLOADS_DIR = os.path.join(DATA_BASE_DIR, "data", "uploads")
SIGNATURES_DIR = os.path.join(DATA_BASE_DIR, "data", "signatures")

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


def analyze_document_content(body: str, language: str = "English", style: str = "layman") -> str:
    lang_prompt = f"Provide the analysis in {language}."
    style_prompt = ""
    if style == "layman":
        style_prompt = (
            "Explain everything in simple, layman terms that someone without a legal background can understand. "
            "Avoid complex legal jargon, and if you must use it, explain it clearly."
        )
    
    system_prompt = (
        "You are an AI assistant specialized in analyzing legal documents, warranties, and agreements. "
        f"Analyze the following document content. {lang_prompt} {style_prompt} "
        "Summarize the key points, identify important obligations, "
        "and highlight any potential risks or notable sections. "
        "Format the output using clear markdown headers and bullet points."
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
    # Check if config is configured, if not, redirect to settings
    if not config.get("openai_api_key") and config.get("use_openai"):
        flash("Please configure your OpenAI API Key in settings.", "warning")
        return redirect(url_for("signature.settings"))
    
    docs = load_docs()
    return render_template("docdashboard.html", docs=docs)


@signature_bp.route("/settings", methods=["GET", "POST"])
def settings():
    global config, USE_OPENAI, OPENAI_API_KEY, OLLAMA_BASE_URL
    
    if request.method == "POST":
        use_openai = request.form.get("use_openai") == "on"
        openai_key = request.form.get("openai_api_key", "").strip()
        ollama_url = request.form.get("ollama_base_url", "http://localhost:11434").strip()
        email = request.form.get("email", "").strip()
        app_password = request.form.get("app_password", "").strip()
        
        config["use_openai"] = use_openai
        config["openai_api_key"] = openai_key
        config["ollama_base_url"] = ollama_url
        config["email"] = email
        config["app_password"] = app_password
        
        # Save to CONFIG_FILE
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            
        # Update current runtime variables
        USE_OPENAI = use_openai
        OPENAI_API_KEY = openai_key
        OLLAMA_BASE_URL = ollama_url
        
        flash("Settings updated successfully.", "success")
        return redirect(url_for("signature.index"))
        
    return render_template("settings.html", config=config)

@signature_bp.route("/doc/send/<doc_id>", methods=["POST"])
def doc_send(doc_id):
    to_email = request.form.get("to")
    cc_email = request.form.get("cc")
    bcc_email = request.form.get("bcc")
    message_text = request.form.get("message")

    if not to_email:
        return jsonify({"error": "Recipient email is required"}), 400

    # Load metadata to find the file
    docs = load_docs()
    doc = next((d for d in docs if d['id'] == doc_id), None)
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    # Determine file path
    file_path = doc.get('file_path')
    if not file_path or not os.path.exists(file_path):
        # Fallback to stored_filename if file_path is missing or incorrect
        stored_filename = doc.get('stored_filename')
        if stored_filename:
            file_path = os.path.join(UPLOADS_DIR, stored_filename)
        else:
            # Last resort fallback
            file_path = os.path.join(UPLOADS_DIR, doc.get('filename'))

    attachment_name = doc.get('filename', 'document.pdf')

    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": f"File not found on disk: {file_path}"}), 404

    # Get email credentials
    sender_email = config.get("email")
    app_password = config.get("app_password")
    if app_password:
        app_password = app_password.replace(" ", "").strip()

    if not sender_email or not app_password:
        return jsonify({"error": "Email credentials not configured in settings"}), 400

    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = to_email
        if cc_email:
            msg['Cc'] = cc_email
        msg['Subject'] = f"Document: {attachment_name}"

        msg.attach(MIMEText(message_text, 'plain'))

        # Attach file
        with open(file_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=attachment_name)
            part['Content-Disposition'] = f'attachment; filename="{attachment_name}"'
            msg.attach(part)

        # Send email
        recipients = [to_email]
        if cc_email:
            recipients.append(cc_email)
        if bcc_email:
            recipients.append(bcc_email)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipients, msg.as_string())

        return jsonify({"success": True, "message": "Email sent successfully!"})

    except Exception as e:
        return jsonify({"error": f"Failed to send email: {str(e)}"}), 500


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

            extension = original_filename.split('.')[-1].lower() if '.' in original_filename else 'pdf'
            
            meta = {
                "id": file_id,
                "original_filename": original_filename,
                "stored_filename": stored_filename,
                "filename": original_filename,
                "file_path": file_path,
                "type": extension,
                "status": "uploaded",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "uploaded_at": datetime.utcnow().isoformat()
            }
            write_metadata(meta)

            flash(f"Document {original_filename} uploaded successfully.", "success")
            return redirect(url_for("signature.index"))

        else:
            flash("Please provide a PDF file URL.", "danger")
            return redirect(url_for("signature.docupload"))

    # GET -> show current docs
    docs = load_docs()
    return render_template("docdashboard.html", docs=docs)


# Serve uploaded files
@signature_bp.route('/docuploaded_file/<path:filename>')
def docuploaded_file(filename):
    # Use global UPLOADS_DIR
    uploads_dir = UPLOADS_DIR

    # Loop through all .json meta files
    for f in os.listdir(uploads_dir):
        if f.endswith(".json"):
            meta_path = os.path.join(uploads_dir, f)
            with open(meta_path) as meta_file:
                meta = json.load(meta_file)

            if meta.get("original_filename") == filename:
                stored_filename = meta.get("stored_filename")
                if stored_filename and os.path.exists(os.path.join(uploads_dir, stored_filename)):
                    return send_from_directory(uploads_dir, stored_filename)

    abort(404, description=f"No stored file found for {filename}")


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

    file_path = meta.get('file_path')
    if not file_path or not os.path.exists(file_path):
        stored_filename = meta.get('stored_filename')
        if stored_filename:
            file_path = os.path.join(UPLOADS_DIR, stored_filename)
        else:
            file_path = os.path.join(UPLOADS_DIR, meta.get('filename'))

    if not file_path or not os.path.exists(file_path):
        flash("Document file not found on disk.", "danger")
        return redirect(url_for("signature.index"))

    return send_file(file_path, as_attachment=True, download_name=meta.get("filename", doc_id))


@signature_bp.route("/save_signed_pdf", methods=["POST"])
def save_signed_pdf():
    data = request.json
    pdf_base64 = data.get("pdf_base64")
    original_filename = data.get("original_filename")

    if not pdf_base64 or not original_filename:
        return {"success": False, "error": "Missing data"}, 400

    file_id = str(uuid.uuid4())
    # generate new signed filename
    signed_filename = f"signed_{file_id}_{original_filename}"
    signed_path = os.path.join(UPLOADS_DIR, signed_filename)

    # decode base64
    pdf_bytes = base64.b64decode(pdf_base64.split(",")[1] if "," in pdf_base64 else pdf_base64)
    with open(signed_path, "wb") as f:
        f.write(pdf_bytes)

    # create meta file
    # create meta file (use only the id in the meta filename)
    meta_path = os.path.join(UPLOADS_DIR, f"signed_{file_id}.json")

    meta = {
        "id": f"signed_{file_id}",
        "original_filename": original_filename,
        "stored_filename": signed_filename,
        "filename": original_filename,
        "file_path": signed_path,   # better to store full path instead of just folder
        "type": "pdf",
        "status": "signed",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"success": True, "filename": signed_filename}


# Analyse document
@signature_bp.route("/analyze_doc/<doc_id>", methods=["GET", "POST"])
def analyze_doc(doc_id):
    meta_path = os.path.join(UPLOADS_DIR, f"{doc_id}.json")
    if not os.path.exists(meta_path):
        flash("Document metadata not found.", "danger")
        return redirect(url_for("signature.index"))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if request.method == "POST":
        language = request.form.get("language", "English")
        style = request.form.get("style", "layman")
        
        file_path = meta.get("file_path")
        if not file_path or not os.path.exists(file_path):
            flash("Document file not found.", "danger")
            return redirect(url_for("signature.index"))

        text = ""
        filename = file_path.lower()
        
        try:
            if filename.endswith(".pdf"):
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
            elif filename.endswith(".docx"):
                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            elif filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                flash("Unsupported file type for analysis.", "danger")
                return redirect(url_for("signature.index"))
        except Exception as e:
            flash(f"Error extracting text: {e}", "danger")
            return redirect(url_for("signature.index"))

        if not text.strip():
            flash("No text content found in document to analyze.", "warning")
            return redirect(url_for("signature.index"))

        try:
            raw_analysis = analyze_document_content(text, language=language, style=style)
            # Convert analysis to HTML for better display
            html_analysis = markdown.markdown(raw_analysis)
            
            # Store analysis in metadata
            meta["analysis"] = html_analysis
            meta["analysis_language"] = language
            meta["analysis_style"] = style
            write_metadata(meta)
            
            return render_template("docanalysis.html", doc=meta, analysis=html_analysis)
        except Exception as e:
            flash(f"Analysis failed: {e}", "danger")
            return redirect(url_for("signature.index"))

    analysis_html = meta.get("analysis")
    
    # Check if JSON is requested (more robust check)
    is_json = request.is_json or \
              'application/json' in request.headers.get('Accept', '') or \
              request.args.get('format') == 'json'

    if is_json:
        if not analysis_html:
            # Fallback to current settings if no analysis exists
            try:
                file_path = meta.get("file_path")
                if not file_path or not os.path.exists(file_path):
                     return jsonify({"success": False, "analysis": "Document file not found."})
                
                text = ""
                # Quick text extract
                filename = file_path.lower()
                if filename.endswith(".pdf"):
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            extracted = page.extract_text()
                            if extracted: text += extracted + "\n"
                elif filename.endswith(".docx"):
                    import docx
                    doc = docx.Document(file_path)
                    for para in doc.paragraphs:
                        text += para.text + "\n"
                elif filename.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()

                if not text.strip():
                    return jsonify({"success": False, "analysis": "Low or no text content found to analyze."})

                analysis_text_raw = analyze_document_content(text)
                analysis_html = markdown.markdown(analysis_text_raw)
                return jsonify({"success": True, "analysis": analysis_html})
            except Exception as e:
                return jsonify({"success": False, "analysis": f"Extraction error: {str(e)}"})
        
        # Strip HTML for the simple modal if needed, or just return as is
        return jsonify({"success": True, "analysis": analysis_html})

    # GET -> show analysis options or current analysis if exists
    return render_template("docanalysis.html", doc=meta, analysis=analysis_html)


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

    # decode and encrypt
    raw_bytes = base64.b64decode(sig_data.split(",")[1])
    encrypted = cipher.encrypt(raw_bytes)

    with open(sig_path, "wb") as f:
        f.write(encrypted)

    # update document JSON
    meta_path = os.path.join(UPLOADS_DIR, f"{doc_id}.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["signature"] = sig_path
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return {"success": True, "signature_url": url_for("signature.serve_signature", sig_id=sig_id)}


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

    # read + decrypt
    with open(sig_path, "rb") as f:
        encrypted = f.read()
    raw_bytes = cipher.decrypt(encrypted)

    return send_file(
        io.BytesIO(raw_bytes),
        mimetype="image/png",
        as_attachment=False,
        download_name=sig_id
    )


@signature_bp.route("/serve/<doc_id>")
def serve_doc(doc_id):
    meta_path = os.path.join(UPLOADS_DIR, f"{doc_id}.json")
    if not os.path.exists(meta_path):
        flash("Document metadata not found.", "danger")
        return redirect(url_for("signature.index"))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    file_path = meta.get('file_path')
    if not file_path or not os.path.exists(file_path):
        stored_filename = meta.get('stored_filename')
        if stored_filename:
            file_path = os.path.join(UPLOADS_DIR, stored_filename)
        else:
            file_path = os.path.join(UPLOADS_DIR, meta.get('filename'))

    if not file_path or not os.path.exists(file_path):
        flash("Document file not found on disk.", "danger")
        return redirect(url_for("signature.index"))

    # Serve PDF inline
    return send_file(file_path, mimetype='application/pdf')


@signature_bp.route("/download_signed/<doc_id>")
def download_signed(doc_id):
    # load your meta JSON
    meta_path = os.path.join(UPLOADS_DIR, f"{doc_id}.json")
    if not os.path.exists(meta_path):
        flash("Document not found", "danger")
        return redirect(url_for("signature.index"))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    signed_path = meta.get("stored_filename")
    if not signed_path or not os.path.exists(os.path.join(UPLOADS_DIR, signed_path)):
        flash("Signed file not found", "danger")
        return redirect(url_for("signature.index"))

    return send_file(
        os.path.join(UPLOADS_DIR, signed_path),
        as_attachment=True,
        download_name=meta.get("filename")
    )
