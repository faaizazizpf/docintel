import os
import json
import re
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------
# 1. Read and Clean Text
# ------------------------------

def read_file(path):
    try:
        if path.lower().endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                return "\n".join([page.extract_text() or "" for page in pdf.pages])
        else:
            return open(path, "r", encoding="utf8").read()
    except:
        return None


def clean_text(text):
    if not text:
        return ""
    return " ".join(text.split())  # removes excess spaces/newlines


# ------------------------------
# 2. Classification Rules
# ------------------------------

def classify_document(text):
    t = text.lower()

    if "invoice" in t or "total amount" in t:
        return "Invoice"

    if "account number" in t or "usage" in t or "amount due" in t:
        return "Utility Bill"

    if ("experience" in t and "email" in t) or ("summary" in t and "phone" in t):
        return "Resume"

    if len(t.strip()) == 0:
        return "Unclassifiable"

    return "Other"


# ------------------------------
# 3. Extraction Functions
# ------------------------------

def extract_invoice(text):
    return {
        "invoice_number": (m.group(2) if (m := re.search(r"(invoice[#:\s]*)(\S+)", text, re.I)) else None),
        "date": (m.group(0) if (m := re.search(r"\d{4}-\d{2}-\d{2}", text)) else None),
        "company": (m.group(1).strip() if (m := re.search(r"company[:\s]*([A-Za-z ]+)", text)) else None),
        "total_amount": (m.group(1) if (m := re.search(r"\$([0-9.,]+)", text)) else None),
    }


def extract_resume(text):
    return {
        "name": text.split("\n")[0].strip(),
        "email": (m.group(0) if (m := re.search(r"[\w\.-]+@[\w\.-]+", text)) else None),
        "phone": (m.group(0) if (m := re.search(r"\+?\d[\d\s-]{8,}", text)) else None),
        "experience_years": (m.group(1) if (m := re.search(r"(\d+)\s+years", text)) else None),
    }


def extract_bill(text):
    return {
        "account_number": (m.group(1) if (m := re.search(r"account number[:\s]*([A-Za-z0-9-]+)", text, re.I)) else None),
        "date": (m.group(0) if (m := re.search(r"\d{4}-\d{2}-\d{2}", text)) else None),
        "usage_kwh": (m.group(1) if (m := re.search(r"(\d+)\s*kwh", text, re.I)) else None),
        "amount_due": (m.group(1) if (m := re.search(r"\$([0-9.,]+)", text)) else None),
    }


# ------------------------------
# 4. Main Document Processing
# ------------------------------

def process_documents(folder="input"):
    results = {}

    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        text = read_file(path)

        if not text:  # unreadable
            results[filename] = {"class": "Unclassifiable"}
            continue

        text_clean = clean_text(text)
        doc_class = classify_document(text_clean)

        if doc_class == "Invoice":
            data = extract_invoice(text_clean)
        elif doc_class == "Resume":
            data = extract_resume(text_clean)
        elif doc_class == "Utility Bill":
            data = extract_bill(text_clean)
        else:
            data = {}

        results[filename] = {
            "class": doc_class,
            "text": text_clean,
            **data
        }

    return results


# ------------------------------
# 5. Semantic Search
# ------------------------------

def build_embeddings(results):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = {}

    for file, data in results.items():
        text = data.get("text", "")
        embeddings[file] = model.encode(text)

    return model, embeddings


def semantic_search(query, model, embeddings, top_k=5):
    q_emb = model.encode(query)
    scores = {}

    for file, emb in embeddings.items():
        scores[file] = float(cosine_similarity([q_emb], [emb])[0][0])

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]


# ------------------------------
# 6. Save result to JSON
# ------------------------------

def save_output(results, filename="all_data.json"):
    cleaned = {file: {k: v for k, v in data.items() if k != "text"} for file, data in results.items()}
    with open(filename, "w") as f:
        json.dump(cleaned, f, indent=2)
    print(f"\nOutput saved to {filename}")


def save_matches_full(matches, results, filename="search_results.json"):
    output = {}

    for file, score in matches:
        # include all fields from original results, EXCEPT the raw text
        entry = {k: v for k, v in results[file].items() if k != "text"}
        entry["similarity_score"] = score  # optional
        output[file] = entry

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSemantic search full results saved to {filename}")


# ------------------------------
# 7. Run Entire Pipeline
# ------------------------------

if __name__ == "__main__":
    print("Processing documents...")
    results = process_documents()

    print("Building embeddings...")
    model, embeddings = build_embeddings(results)

    save_output(results)

    print("\nSemantic Search Example:")
    query = "Find all documents mentioning payments due in January"
    print(f"Query: {query}")

    matches = semantic_search(query, model, embeddings)

    print("\nTop Matches:")
    for file, score in matches:
        print(f"{file} — score: {score:.4f}")

    # Save full JSON of matched documents
    save_matches_full(matches, results, "output.json")

    print("\nDone.")
