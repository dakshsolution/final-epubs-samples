#!/usr/bin/env python3
"""Extract EPUB files and build HTML index/split-view pages."""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, unquote, urlsplit
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import pypdfium2 as pdfium

HTML_MEDIA_TYPES = {
    "application/xhtml+xml",
    "text/html",
    "application/xml",
}


@dataclass
class ManifestItem:
    item_id: str
    href: str
    media_type: str
    properties: str


@dataclass
class LinkEntry:
    title: str
    target: Path  # path relative to project root
    fragment: str | None = None


@dataclass
class BookRecord:
    epub_file: Path
    title: str
    toc_links: list[LinkEntry]
    chapter_links: list[LinkEntry]
    split_page: Path
    preview_image: Path | None
    pdf_file: Path | None


def normalize_key(name: str) -> str:
    name = re.sub(r"_epub$", "", name, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def encode_rel_href(from_dir: Path, to_path: Path, fragment: str | None = None) -> str:
    rel_path = os.path.relpath(to_path, start=from_dir)
    href = quote(Path(rel_path).as_posix(), safe="/-._~")
    if fragment:
        href += "#" + quote(fragment, safe="-._~")
    return href


def resolve_local_href(base_dir: Path, href: str) -> tuple[Path | None, str | None]:
    parts = urlsplit(href)
    if parts.scheme or parts.netloc:
        return None, None

    fragment = parts.fragment or None
    path_part = unquote(parts.path)
    if not path_part:
        return base_dir.resolve(strict=False), fragment

    resolved = (base_dir / Path(path_part)).resolve(strict=False)
    return resolved, fragment


def read_opf_path(extracted_book_dir: Path) -> Path:
    container_xml = extracted_book_dir / "META-INF" / "container.xml"
    if not container_xml.exists():
        raise FileNotFoundError(f"Missing container.xml: {container_xml}")

    root = ET.parse(container_xml).getroot()
    rootfile = root.find(".//{*}rootfile")
    if rootfile is None:
        raise ValueError(f"No rootfile in {container_xml}")

    opf_rel = rootfile.attrib.get("full-path", "").strip()
    if not opf_rel:
        raise ValueError(f"No OPF path in {container_xml}")

    opf_path = extracted_book_dir / Path(opf_rel)
    if not opf_path.exists():
        raise FileNotFoundError(f"OPF not found: {opf_path}")

    return opf_path


def parse_opf(opf_path: Path) -> tuple[str, dict[str, ManifestItem], list[str], str | None, Path]:
    root = ET.parse(opf_path).getroot()

    title = ""
    for title_node in root.findall(".//{*}metadata/{*}title"):
        text = "".join(title_node.itertext()).strip()
        if text:
            title = text
            break

    manifest: dict[str, ManifestItem] = {}
    for item in root.findall(".//{*}manifest/{*}item"):
        item_id = item.attrib.get("id", "").strip()
        href = item.attrib.get("href", "").strip()
        media_type = item.attrib.get("media-type", "").strip()
        properties = item.attrib.get("properties", "").strip()
        if item_id and href:
            manifest[item_id] = ManifestItem(item_id, href, media_type, properties)

    spine_ids: list[str] = []
    spine_toc_id: str | None = None
    spine = root.find(".//{*}spine")
    if spine is not None:
        spine_toc_id = spine.attrib.get("toc")
        for itemref in spine.findall("{*}itemref"):
            idref = itemref.attrib.get("idref", "").strip()
            if idref:
                spine_ids.append(idref)

    return title, manifest, spine_ids, spine_toc_id, opf_path.parent


def gather_chapters(
    manifest: dict[str, ManifestItem],
    spine_ids: Iterable[str],
    opf_dir: Path,
    extracted_book_dir: Path,
    extracted_root: Path,
) -> list[LinkEntry]:
    chapters: list[LinkEntry] = []
    seen: set[Path] = set()

    for item_id in spine_ids:
        item = manifest.get(item_id)
        if not item:
            continue
        if item.media_type.lower() not in HTML_MEDIA_TYPES:
            continue

        target_abs = (opf_dir / Path(unquote(urlsplit(item.href).path))).resolve(strict=False)
        if not target_abs.exists():
            continue
        if target_abs in seen:
            continue
        seen.add(target_abs)

        try:
            rel_to_book = target_abs.relative_to(extracted_book_dir)
        except ValueError:
            continue

        chapter_name = rel_to_book.name
        chapters.append(
            LinkEntry(
                title=chapter_name,
                target=(extracted_root / extracted_book_dir.name / rel_to_book),
            )
        )

    return chapters


def parse_nav_toc(
    nav_file: Path,
    extracted_book_dir: Path,
    extracted_root: Path,
) -> list[LinkEntry]:
    text = nav_file.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(text, "xml")

    toc_nav = soup.find("nav", attrs={"epub:type": "toc"})
    if toc_nav is None:
        toc_nav = soup.find("nav")
    if toc_nav is None:
        return []

    links: list[LinkEntry] = []
    for anchor in toc_nav.find_all("a"):
        href = (anchor.get("href") or "").strip()
        if not href:
            continue

        target_abs, fragment = resolve_local_href(nav_file.parent, href)
        if not target_abs:
            continue

        try:
            rel_to_book = target_abs.relative_to(extracted_book_dir)
        except ValueError:
            continue

        label = anchor.get_text(" ", strip=True) or rel_to_book.name
        links.append(
            LinkEntry(
                title=label,
                target=(extracted_root / extracted_book_dir.name / rel_to_book),
                fragment=fragment,
            )
        )

    return links


def parse_ncx_toc(
    ncx_file: Path,
    extracted_book_dir: Path,
    extracted_root: Path,
) -> list[LinkEntry]:
    text = ncx_file.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(text, "xml")

    links: list[LinkEntry] = []
    for nav_point in soup.find_all("navPoint"):
        label_node = nav_point.find("text")
        label = label_node.get_text(" ", strip=True) if label_node else ""

        content = nav_point.find("content")
        href = (content.get("src") or "").strip() if content else ""
        if not href:
            continue

        target_abs, fragment = resolve_local_href(ncx_file.parent, href)
        if not target_abs:
            continue

        try:
            rel_to_book = target_abs.relative_to(extracted_book_dir)
        except ValueError:
            continue

        links.append(
            LinkEntry(
                title=label or rel_to_book.name,
                target=(extracted_root / extracted_book_dir.name / rel_to_book),
                fragment=fragment,
            )
        )

    return links


def gather_toc_links(
    manifest: dict[str, ManifestItem],
    spine_toc_id: str | None,
    opf_dir: Path,
    extracted_book_dir: Path,
    extracted_root: Path,
) -> list[LinkEntry]:
    nav_candidates: list[Path] = []
    ncx_candidates: list[Path] = []

    for item in manifest.values():
        item_path = (opf_dir / Path(unquote(urlsplit(item.href).path))).resolve(strict=False)
        props = set(item.properties.lower().split())

        if "nav" in props or item.href.lower().endswith("toc.xhtml"):
            nav_candidates.append(item_path)

        if item.item_id == spine_toc_id or item.media_type.lower() == "application/x-dtbncx+xml":
            ncx_candidates.append(item_path)

    for nav_file in nav_candidates:
        if nav_file.exists():
            links = parse_nav_toc(nav_file, extracted_book_dir, extracted_root)
            if links:
                return links

    for ncx_file in ncx_candidates:
        if ncx_file.exists():
            links = parse_ncx_toc(ncx_file, extracted_book_dir, extracted_root)
            if links:
                return links

    return []


def find_matching_pdf(epub_file: Path, pdf_files: list[Path]) -> Path | None:
    epub_stem = re.sub(r"_epub$", "", epub_file.stem, flags=re.IGNORECASE)

    exact_map = {pdf.stem.lower(): pdf for pdf in pdf_files}
    exact = exact_map.get(epub_stem.lower())
    if exact:
        return exact

    norm_map = {normalize_key(pdf.stem): pdf for pdf in pdf_files}
    return norm_map.get(normalize_key(epub_stem))


def render_first_pdf_page(pdf_path: Path, output_path: Path) -> bool:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf = pdfium.PdfDocument(str(pdf_path))
        page = pdf[0]
        pil_image = page.render(scale=1.8).to_pil()
        pil_image.save(output_path)
        page.close()
        pdf.close()
        return True
    except Exception:
        return False


def link_list_html(links: list[LinkEntry], base_dir: Path, empty_label: str) -> str:
    if not links:
        return f"<span class=\"empty\">{html.escape(empty_label)}</span>"

    parts = ["<ul>"]
    for link in links:
        href = encode_rel_href(base_dir, link.target, link.fragment)
        title = html.escape(link.title)
        parts.append(f'<li><a href="{href}" target="_blank">{title}</a></li>')
    parts.append("</ul>")
    return "".join(parts)


def write_split_page(book: BookRecord, split_file: Path) -> None:
    split_dir = split_file.parent
    first_link = book.chapter_links[0] if book.chapter_links else (book.toc_links[0] if book.toc_links else None)
    initial_epub_href = encode_rel_href(split_dir, first_link.target, first_link.fragment) if first_link else ""

    toc_html_items = []
    for link in book.toc_links:
        href = encode_rel_href(split_dir, link.target, link.fragment)
        toc_html_items.append(f'<li><a href="{href}" target="epubPane">{html.escape(link.title)}</a></li>')

    chapter_html_items = []
    for link in book.chapter_links:
        href = encode_rel_href(split_dir, link.target, link.fragment)
        chapter_html_items.append(f'<li><a href="{href}" target="epubPane">{html.escape(link.title)}</a></li>')

    if book.pdf_file:
        pdf_panel = f'<iframe name="pdfPane" src="{encode_rel_href(split_dir, book.pdf_file)}" title="PDF"></iframe>'
    else:
        pdf_panel = '<div class="missing">No matching PDF found for this EPUB.</div>'

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(book.title)} - Split View</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Tahoma, sans-serif; background: #f5f7fb; color: #1f2937; }}
    .header {{ padding: 12px 16px; background: #0f172a; color: white; display: flex; justify-content: space-between; gap: 10px; }}
    .header a {{ color: #93c5fd; text-decoration: none; }}
    .layout {{ display: grid; grid-template-columns: 280px 1fr; min-height: calc(100vh - 52px); }}
    .sidebar {{ border-right: 1px solid #d1d5db; padding: 12px; overflow: auto; background: #ffffff; }}
    .sidebar h3 {{ margin: 12px 0 6px; font-size: 14px; }}
    .sidebar ul {{ margin: 0; padding-left: 18px; }}
    .sidebar li {{ margin: 4px 0; line-height: 1.35; }}
    .sidebar a {{ color: #0a3ea3; text-decoration: none; }}
    .panes {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
    iframe {{ width: 100%; height: calc(100vh - 52px); border: none; background: white; }}
    .missing {{ display: flex; align-items: center; justify-content: center; font-weight: 600; color: #b91c1c; background: #fee2e2; }}
    @media (max-width: 980px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ max-height: 42vh; border-right: none; border-bottom: 1px solid #d1d5db; }}
      .panes {{ grid-template-columns: 1fr; }}
      iframe {{ height: 58vh; }}
    }}
  </style>
</head>
<body>
  <div class="header">
    <strong>{html.escape(book.title)}</strong>
    <a href="{encode_rel_href(split_dir, Path('index.html'))}">Back to index</a>
  </div>
  <div class="layout">
    <aside class="sidebar">
      <h3>TOC Links</h3>
      <ul>{''.join(toc_html_items) if toc_html_items else '<li>No TOC links found.</li>'}</ul>
      <h3>Chapter XHTML Links</h3>
      <ul>{''.join(chapter_html_items) if chapter_html_items else '<li>No chapter XHTML files found.</li>'}</ul>
    </aside>
    <section class="panes">
      <iframe name="epubPane" src="{initial_epub_href}" title="EPUB HTML"></iframe>
      {pdf_panel}
    </section>
  </div>
</body>
</html>
"""

    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(html_text, encoding="utf-8")


def write_index(index_file: Path, books: list[BookRecord]) -> None:
    rows: list[str] = []

    for i, book in enumerate(books, start=1):
        toc_html = link_list_html(book.toc_links, Path("."), "No TOC entries")
        chapter_html = link_list_html(book.chapter_links, Path("."), "No chapter XHTML files")

        preview_html = (
            f'<img src="{encode_rel_href(Path("."), book.preview_image)}" alt="{html.escape(book.title)} preview" />'
            if book.preview_image
            else '<span class="empty">No preview</span>'
        )

        pdf_link_html = (
            f'<a href="{encode_rel_href(Path("."), book.pdf_file)}" target="_blank">{html.escape(book.pdf_file.name)}</a>'
            if book.pdf_file
            else '<span class="empty">No matching PDF</span>'
        )

        rows.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{i}</td>",
                    f"<td><strong>{html.escape(book.title)}</strong><br><small>{html.escape(book.epub_file.name)}</small></td>",
                    f"<td class=\"preview\">{preview_html}</td>",
                    f"<td class=\"links\">{toc_html}</td>",
                    f"<td class=\"links\">{chapter_html}</td>",
                    f"<td><a href=\"{encode_rel_href(Path('.'), book.split_page)}\">Open split view</a><br>{pdf_link_html}</td>",
                    "</tr>",
                ]
            )
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EPUB Index</title>
  <style>
    body {{ margin: 0; padding: 20px; font-family: Segoe UI, Tahoma, sans-serif; background: #f3f4f6; color: #111827; }}
    h1 {{ margin: 0 0 8px; }}
    p {{ margin: 0 0 16px; color: #374151; }}
    .wrap {{ overflow-x: auto; border: 1px solid #d1d5db; background: #ffffff; border-radius: 10px; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 1200px; }}
    th, td {{ border: 1px solid #e5e7eb; text-align: left; vertical-align: top; padding: 10px; font-size: 14px; }}
    th {{ background: #111827; color: #ffffff; position: sticky; top: 0; }}
    .links {{ max-width: 300px; }}
    .links ul {{ margin: 0; padding-left: 18px; max-height: 190px; overflow: auto; }}
    .links li {{ margin: 4px 0; }}
    .links a {{ color: #0a3ea3; text-decoration: none; }}
    .preview img {{ width: 140px; border: 1px solid #d1d5db; border-radius: 5px; display: block; }}
    .empty {{ color: #6b7280; font-style: italic; }}
  </style>
</head>
<body>
  <h1>EPUB Extraction Index</h1>
  <p>Total EPUBs processed: {len(books)}</p>
  <div class="wrap">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>EPUB</th>
          <th>First Page Image</th>
          <th>TOC Links</th>
          <th>Chapter XHTML Links</th>
          <th>Split View (HTML + PDF)</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""

    index_file.write_text(html_text, encoding="utf-8")


def process_books(args: argparse.Namespace) -> list[BookRecord]:
    root = Path(args.root).resolve()
    epub_dir = (root / args.epub_dir).resolve()
    pdf_dir = (root / args.pdf_dir).resolve()

    extracted_root = Path(args.extracted_dir)
    preview_root = Path(args.preview_dir)
    split_root = Path(args.split_dir)

    (root / extracted_root).mkdir(parents=True, exist_ok=True)
    (root / preview_root).mkdir(parents=True, exist_ok=True)
    (root / split_root).mkdir(parents=True, exist_ok=True)

    epub_files = sorted(epub_dir.glob("*.epub"), key=lambda p: p.name.lower())
    pdf_files = sorted(pdf_dir.glob("*.pdf"), key=lambda p: p.name.lower()) if pdf_dir.exists() else []

    books: list[BookRecord] = []

    for epub in epub_files:
        book_extract_dir = (root / extracted_root / epub.stem).resolve()
        if book_extract_dir.exists():
            shutil.rmtree(book_extract_dir)
        book_extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(epub, "r") as zf:
            zf.extractall(book_extract_dir)

        opf_path = read_opf_path(book_extract_dir)
        title, manifest, spine_ids, spine_toc_id, opf_dir = parse_opf(opf_path)

        toc_links = gather_toc_links(manifest, spine_toc_id, opf_dir, book_extract_dir, extracted_root)
        chapter_links = gather_chapters(manifest, spine_ids, opf_dir, book_extract_dir, extracted_root)

        pdf_match_abs = find_matching_pdf(epub, pdf_files)
        pdf_match_rel = pdf_match_abs.relative_to(root) if pdf_match_abs else None

        preview_rel: Path | None = None
        if pdf_match_abs:
            out_preview = root / preview_root / f"{epub.stem}.png"
            if render_first_pdf_page(pdf_match_abs, out_preview):
                preview_rel = out_preview.relative_to(root)

        split_rel = split_root / f"{epub.stem}.html"

        book = BookRecord(
            epub_file=epub.relative_to(root),
            title=title or epub.stem,
            toc_links=toc_links,
            chapter_links=chapter_links,
            split_page=split_rel,
            preview_image=preview_rel,
            pdf_file=pdf_match_rel,
        )

        write_split_page(book, root / split_rel)
        books.append(book)

    return books


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract all EPUBs and generate index/split-view HTML files."
    )
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--epub-dir", default=".", help="Directory containing .epub files")
    parser.add_argument("--pdf-dir", default="vendor_pdfs", help="Directory containing PDF files")
    parser.add_argument("--extracted-dir", default="extracted_epubs", help="Output folder for extracted EPUB files")
    parser.add_argument("--preview-dir", default="preview_images", help="Output folder for first-page preview images")
    parser.add_argument("--split-dir", default="split_views", help="Output folder for split-view HTML pages")
    parser.add_argument("--index-file", default="index.html", help="Output index HTML file name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    books = process_books(args)
    index_file = root / args.index_file
    write_index(index_file, books)

    print(f"Processed {len(books)} EPUB files.")
    print(f"Index written to: {index_file}")


if __name__ == "__main__":
    main()
