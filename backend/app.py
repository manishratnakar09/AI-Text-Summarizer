"""Flask API + static frontend server for the AI Text Summarizer."""
import io
import os
import time
from collections import defaultdict, deque
from xml.sax.saxutils import escape

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", ".env"))

import ai_handler                                     # noqa: E402  (after load_dotenv)
from extractor import ExtractionError, clean_text, extract_text   # noqa: E402
from summarizer import summarize, word_count          # noqa: E402

FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = (MAX_UPLOAD_MB + 1) * 1024 * 1024
app.config["JSON_AS_ASCII"] = False

_hits = defaultdict(deque)


def rate_limited():
    """Tiny in-memory per-IP limiter (protects your AI quota)."""
    now, q = time.time(), _hits[request.remote_addr]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


def error(message, status=400):
    return jsonify({"error": message}), status


@app.after_request
def security_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "ai_configured": ai_handler.is_configured(),
                    "provider": ai_handler.provider(), "max_upload_mb": MAX_UPLOAD_MB})


@app.route("/api/extract", methods=["POST"])
def api_extract():
    f = request.files.get("file")
    if f is None:
        return error("No file uploaded.")
    try:
        text = extract_text(f.filename or "", f.read(), MAX_UPLOAD_MB * 1024 * 1024)
    except ExtractionError as exc:
        return error(str(exc))
    return jsonify({"text": text, "filename": f.filename, "word_count": word_count(text)})


@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    if rate_limited():
        return error("Too many requests. Please wait a minute and try again.", 429)

    if request.files.get("file"):                                   # multipart upload
        src, data = request.files["file"], request.form
        try:
            text = extract_text(src.filename or "", src.read(), MAX_UPLOAD_MB * 1024 * 1024)
        except ExtractionError as exc:
            return error(str(exc))
    else:                                                           # JSON or form text
        data = request.get_json(silent=True) or request.form
        text = clean_text(str(data.get("text", "")))

    truthy = str(data.get("bullets", "false")).lower() in ("1", "true", "yes", "on")
    try:
        result = summarize(
            text,
            mode=str(data.get("mode", "hybrid")),
            length=str(data.get("length", "medium")),
            custom_type=(data.get("custom_type") or None),
            custom_value=data.get("custom_value"),
            bullets=truthy,
        )
    except ValueError as exc:
        return error(str(exc))
    return jsonify(result)


@app.route("/api/download", methods=["POST"])
def api_download():
    data = request.get_json(silent=True) or {}
    summary = str(data.get("summary", "")).strip()
    fmt = str(data.get("format", "txt")).lower()
    if not summary:
        return error("There is no summary to download.")
    if len(summary) > 100_000:
        return error("Summary is too large to download.")
    if fmt == "txt":
        return Response(summary, mimetype="text/plain; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="summary.txt"'})
    if fmt == "pdf":
        return Response(build_pdf(summary), mimetype="application/pdf",
                        headers={"Content-Disposition": 'attachment; filename="summary.pdf"'})
    return error("Format must be txt or pdf.")


def build_pdf(summary: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm,
                            topMargin=20 * mm, bottomMargin=20 * mm, title="Summary")
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=11, leading=17, spaceAfter=6)
    story = [Paragraph("AI Text Summary", styles["Title"]), Spacer(1, 6)]
    for line in summary.splitlines():
        if line.strip():
            story.append(Paragraph(escape(line.strip()), body))     # escape = no markup injection
    doc.build(story)
    return buf.getvalue()


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return error(f"File is too large. Maximum allowed size is {MAX_UPLOAD_MB} MB.", 413)


@app.errorhandler(404)
def not_found(_):
    return error("Not found.", 404)


@app.errorhandler(405)
def bad_method(_):
    return error("Method not allowed.", 405)


@app.errorhandler(Exception)
def unexpected(exc):
    app.logger.exception("Unhandled error")
    return error("Something went wrong on the server. Please try again.", 500)


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")),
            debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
