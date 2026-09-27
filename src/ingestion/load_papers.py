from pathlib import Path
from pypdf import PdfReader


PAPERS_DIR = Path("data/papers")


def load_paper(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        pages.append({
            "source": pdf_path.name,
            "page": page_number,
            "text": text,
        })

    return pages


def load_all_papers():
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    all_pages = []

    for pdf_path in pdf_files:
        all_pages.extend(load_paper(pdf_path))

    return all_pages


def main():
    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files.\n")

    total_pages = 0

    for pdf_path in pdf_files:
        pages = load_paper(pdf_path)
        total_pages += len(pages)

        text_length = sum(len(page["text"]) for page in pages)

        print(f"{pdf_path.name}")
        print(f"  Pages: {len(pages)}")
        print(f"  Extracted characters: {text_length:,}")

        if pages:
            preview = pages[0]["text"][:300].replace("\n", " ")
            print(f"  Preview: {preview}...")

        print()

    print(f"Total pages: {total_pages}")


if __name__ == "__main__":
    main()
