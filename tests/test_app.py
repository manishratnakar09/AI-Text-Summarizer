"""Automated tests (standard-library unittest).  Run:  python -m unittest discover -s tests -v"""
import io
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))
os.environ["GEMINI_API_KEY"] = ""          # tests must never call a real AI API
os.environ["RATE_LIMIT_PER_MINUTE"] = "100000"

import ai_handler                          # noqa: E402
import app as app_module                   # noqa: E402
import extractor                           # noqa: E402
import summarizer                          # noqa: E402

ARTICLES = os.path.join(ROOT, "samples", "articles")


def read_sample(name):
    with open(os.path.join(ARTICLES, name), encoding="utf-8") as f:
        return f.read()


def make_pdf(text):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, SimpleDocTemplate
    from reportlab.lib.styles import getSampleStyleSheet
    buf = io.BytesIO()
    style = getSampleStyleSheet()["BodyText"]
    SimpleDocTemplate(buf, pagesize=A4).build([Paragraph(p, style) for p in text.split("\n\n")])
    return buf.getvalue()


def make_docx(text):
    import docx
    d = docx.Document()
    for p in text.split("\n\n"):
        d.add_paragraph(p)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


class SummarizerTests(unittest.TestCase):
    def test_empty_text(self):
        with self.assertRaises(ValueError):
            summarizer.summarize("   ")

    def test_very_short_text(self):
        with self.assertRaises(ValueError):
            summarizer.summarize("This is far too short to summarize.")

    def test_summary_is_shorter_and_extractive(self):
        text = read_sample("news_article.txt")
        r = summarizer.summarize(text, mode="extractive", length="short")
        self.assertLess(r["summary_words"], r["original_words"])
        for s in summarizer.split_sentences(r["summary"]):
            self.assertIn(s, text)                       # no invented sentences

    def test_key_facts_survive(self):
        r = summarizer.summarize(read_sample("news_article.txt"), mode="extractive", length="medium")
        self.assertEqual(summarizer.unsupported_numbers(read_sample("news_article.txt"), r["summary"]), set())

    def test_bullets(self):
        r = summarizer.summarize(read_sample("technical_doc.txt"), mode="extractive", bullets=True)
        self.assertTrue(all(l.startswith("\u2022 ") for l in r["summary"].splitlines()))

    def test_custom_sentences(self):
        r = summarizer.summarize(read_sample("story_essay.txt"), mode="extractive",
                                 custom_type="sentences", custom_value=3)
        self.assertEqual(len(summarizer.split_sentences(r["summary"])), 3)

    def test_custom_words_and_validation(self):
        r = summarizer.summarize(read_sample("academic_paper.txt"), mode="extractive",
                                 custom_type="words", custom_value=60)
        self.assertLessEqual(r["summary_words"], 60 * 1.5)
        with self.assertRaises(ValueError):
            summarizer.summarize(read_sample("academic_paper.txt"), custom_type="words", custom_value=5)
        with self.assertRaises(ValueError):
            summarizer.summarize(read_sample("academic_paper.txt"), custom_type="sentences", custom_value="abc")

    def test_long_document_chunking(self):
        text = read_sample("long_document.txt")
        self.assertGreater(len(summarizer.chunk_text(text)), 5)
        r = summarizer.summarize(text, mode="extractive", length="long")
        self.assertLess(r["summary_words"], 800)

    def test_symbols_only_rejected(self):
        with self.assertRaises(ValueError):
            summarizer.summarize("!@#$%^&*() " * 40)

    def test_unpunctuated_text_is_shortened(self):
        text = " ".join(f"token{i % 37} alpha beta" for i in range(300))
        r = summarizer.summarize(text, mode="extractive", length="short")
        self.assertLess(r["summary_words"], r["original_words"] / 2)

    def test_no_duplicate_sentences_across_chunks(self):
        r = summarizer.summarize(read_sample("long_document.txt"), mode="extractive", length="medium")
        sents = summarizer.split_sentences(r["summary"])
        self.assertEqual(len(sents), len(set(sents)))

    def test_abbreviations_not_split(self):
        s = summarizer.split_sentences("Dr. Rao met Mr. Singh at 5 p.m. today. They talked for an hour.")
        self.assertEqual(len(s), 2)

    def test_hindi_sentence_split(self):
        self.assertEqual(len(summarizer.split_sentences("यह पहला वाक्य है। यह दूसरा वाक्य है। यह तीसरा वाक्य है।")), 3)

    def test_ai_unavailable_falls_back(self):
        r = summarizer.summarize(read_sample("news_article.txt"), mode="hybrid")   # no key configured
        self.assertEqual(r["method"], "extractive")
        self.assertTrue(r["warnings"])

    def test_ai_success_path(self):
        with mock.patch.object(ai_handler, "summarize_with_ai",
                               return_value="The council approved an 18-kilometre metro line costing 4,500 crore rupees."):
            r = summarizer.summarize(read_sample("news_article.txt"), mode="hybrid")
        self.assertEqual(r["method"], "hybrid")
        self.assertEqual(r["warnings"], [])

    def test_ai_hallucinated_number_rejected(self):
        with mock.patch.object(ai_handler, "summarize_with_ai", return_value="The line will cost 9,999 crore rupees."):
            r = summarizer.summarize(read_sample("news_article.txt"), mode="abstractive")
        self.assertEqual(r["method"], "extractive")
        self.assertIn("9999", r["warnings"][0])

    def test_ai_failure_falls_back(self):
        with mock.patch.object(ai_handler, "summarize_with_ai", side_effect=ai_handler.AIUnavailable("API down.")):
            r = summarizer.summarize(read_sample("news_article.txt"), mode="abstractive")
        self.assertEqual(r["method"], "extractive")


class ExtractorTests(unittest.TestCase):
    TEXT = "First paragraph here.\n\nSecond paragraph here."

    def test_txt_encodings(self):
        self.assertIn("café", extractor.extract_text("a.txt", "café au lait".encode("cp1252")))
        self.assertIn("नमस्ते", extractor.extract_text("a.txt", "नमस्ते दुनिया".encode("utf-8")))

    def test_pdf_and_docx(self):
        self.assertIn("Second paragraph", extractor.extract_text("a.pdf", make_pdf(self.TEXT)))
        self.assertIn("Second paragraph", extractor.extract_text("a.docx", make_docx(self.TEXT)))

    def test_rejections(self):
        for name, data in [("a.exe", b"MZ"), ("a.pdf", b"not a pdf"), ("a.docx", b"not zip"), ("a.txt", b"")]:
            with self.assertRaises(extractor.ExtractionError):
                extractor.extract_text(name, data)
        with self.assertRaises(extractor.ExtractionError):
            extractor.extract_text("a.txt", b"x" * 2000, max_bytes=1000)
        with self.assertRaises(extractor.ExtractionError):
            extractor.extract_text("a.docx", b"PK\x03\x04garbage")


class ApiTests(unittest.TestCase):
    def setUp(self):
        app_module.app.config["TESTING"] = False        # keep real error handlers active
        self.c = app_module.app.test_client()
        self.text = read_sample("news_article.txt")

    def post(self, **payload):
        return self.c.post("/api/summarize", json=payload)

    def test_index_and_health(self):
        self.assertEqual(self.c.get("/").status_code, 200)
        self.assertEqual(self.c.get("/style.css").status_code, 200)
        self.assertEqual(self.c.get("/api/health").get_json()["status"], "ok")

    def test_ok(self):
        r = self.post(text=self.text, mode="extractive", length="short")
        self.assertEqual(r.status_code, 200)
        self.assertIn("summary", r.get_json())

    def test_empty_short_invalid(self):
        self.assertEqual(self.post(text="").status_code, 400)
        self.assertEqual(self.post(text="too short").status_code, 400)
        self.assertEqual(self.post(text=self.text, length="huge").status_code, 400)
        self.assertEqual(self.post(text=self.text, mode="magic").status_code, 400)

    def test_file_upload_all_types(self):
        for name, data in [("n.txt", self.text.encode()), ("n.pdf", make_pdf(self.text)), ("n.docx", make_docx(self.text))]:
            r = self.c.post("/api/summarize", data={"file": (io.BytesIO(data), name), "mode": "extractive"},
                            content_type="multipart/form-data")
            self.assertEqual(r.status_code, 200, name)

    def test_upload_rejections(self):
        r = self.c.post("/api/extract", data={"file": (io.BytesIO(b"MZ"), "virus.exe")}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        r = self.c.post("/api/extract", data={"file": (io.BytesIO(b"x" * (12 * 1024 * 1024)), "big.txt")},
                        content_type="multipart/form-data")
        self.assertEqual(r.status_code, 413)

    def test_download(self):
        self.assertEqual(self.c.post("/api/download", json={"summary": "Hi <b>there</b> & you", "format": "txt"}).status_code, 200)
        pdf = self.c.post("/api/download", json={"summary": "Hi <b>there</b> & you", "format": "pdf"})
        self.assertEqual(pdf.status_code, 200)
        self.assertTrue(pdf.data.startswith(b"%PDF"))
        self.assertEqual(self.c.post("/api/download", json={"summary": "", "format": "pdf"}).status_code, 400)
        self.assertEqual(self.c.post("/api/download", json={"summary": "x", "format": "exe"}).status_code, 400)

    def test_no_secret_in_frontend(self):
        for f in ("index.html", "script.js", "style.css"):
            with open(os.path.join(ROOT, "frontend", f), encoding="utf-8") as fh:
                body = fh.read().lower()
            self.assertNotIn("api_key", body)
            self.assertNotIn("aiza", body)


if __name__ == "__main__":
    unittest.main()
