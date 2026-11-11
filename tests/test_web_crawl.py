import json

from lib import web_crawl, web_scrape

HTML = """
<div class="elementor">
  <a href="/tentang-fakultas/">Tentang Fakultas</a>
  <a href="https://ftmm.unair.ac.id/sejarah-fakultas-2#visi">Sejarah</a>
  <a href="/en/about/">English</a>
  <a href="https://example.com/">Luar</a>
  <a href="/wp-content/uploads/2024/pedoman.pdf">Pedoman</a>
  <a href="mailto:ftmm@unair.ac.id">Email</a>
  <a href="/tentang-fakultas/">Duplikat</a>
  <img src="/wp-content/uploads/2024/struktur.png" alt="Struktur Organisasi">
  <img data-src="/wp-content/uploads/2024/pimpinan.webp" alt="Pimpinan">
</div>
"""


def test_classify_urls():
    assert web_crawl.classify("https://ftmm.unair.ac.id/tentang-fakultas/") == "page"
    assert web_crawl.classify("https://ftmm.unair.ac.id/en/about/") == "language"
    assert web_crawl.classify("https://example.com/") == "external"
    assert web_crawl.classify("https://ftmm.unair.ac.id/wp-admin/") == "skip"
    assert web_crawl.classify("https://ftmm.unair.ac.id/wp-content/uploads/a.pdf") == "pdf"
    assert web_crawl.classify("https://ftmm.unair.ac.id/wp-content/uploads/a.png") == "image"


def test_extract_links_resolves_and_dedupes():
    links = web_crawl.extract_links(HTML, "https://ftmm.unair.ac.id/pimpinan-fakultas/")
    urls = [link["url"] for link in links]
    assert "https://ftmm.unair.ac.id/tentang-fakultas/" in urls
    assert "https://ftmm.unair.ac.id/sejarah-fakultas-2" in urls
    assert urls.count("https://ftmm.unair.ac.id/tentang-fakultas/") == 1
    assert all("mailto:" not in url for url in urls)
    tentang = next(link for link in links if link["url"].endswith("/tentang-fakultas/"))
    assert tentang["teks"] == "Tentang Fakultas"
    assert tentang["jenis"] == "page"
    assert next(link for link in links if link["url"].endswith("pedoman.pdf"))["jenis"] == "pdf"


def test_extract_images_handles_data_src():
    images = web_crawl.extract_images(HTML, "https://ftmm.unair.ac.id/pimpinan-fakultas/")
    urls = {image["url"] for image in images}
    assert "https://ftmm.unair.ac.id/wp-content/uploads/2024/struktur.png" in urls
    assert "https://ftmm.unair.ac.id/wp-content/uploads/2024/pimpinan.webp" in urls
    assert next(image for image in images if image["url"].endswith("struktur.png"))["alt"] == "Struktur Organisasi"


def test_needs_render():
    assert web_crawl.needs_render("pendek")
    assert web_crawl.needs_render("x" * 900, "<table class='dataTable'>")
    assert web_crawl.needs_render("x" * 900, "admin-ajax.php")
    assert not web_crawl.needs_render("x" * 900, "<p>teks biasa</p>")


def test_parse_sitemap_index_and_urlset():
    index = b"""<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://ftmm.unair.ac.id/page-sitemap.xml</loc></sitemap>
    </sitemapindex>"""
    urls, children = web_scrape.parse_sitemap(index)
    assert urls == []
    assert children == ["https://ftmm.unair.ac.id/page-sitemap.xml"]

    urlset = b"""<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
      <url><loc>https://ftmm.unair.ac.id/tentang-fakultas/</loc>
        <image:image><image:loc>https://ftmm.unair.ac.id/a.png</image:loc></image:image>
      </url>
      <url><loc>https://ftmm.unair.ac.id/sejarah-fakultas-2/</loc></url>
    </urlset>"""
    urls, children = web_scrape.parse_sitemap(urlset)
    assert urls == [
        "https://ftmm.unair.ac.id/tentang-fakultas/",
        "https://ftmm.unair.ac.id/sejarah-fakultas-2/",
    ]
    assert children == []


def test_merge_dump_replaces_thin_page(tmp_path):
    out = tmp_path / "web_kb"
    out.mkdir()
    thin = {
        "url": "https://ftmm.unair.ac.id/staf-dosen-prodi-teknik-elektro/",
        "tipe": "page",
        "id": 10,
        "slug": "staf-dosen-prodi-teknik-elektro",
        "judul": "Staf Dosen",
        "teks": "Memuat...",
        "jumlah_karakter": 9,
        "status": "ok",
        "links": [],
        "gambar": [],
    }
    (out / "pages.jsonl").write_text(json.dumps(thin) + "\n", encoding="utf-8")

    dump_dir = tmp_path / "raw_browser"
    dump_dir.mkdir()
    dump = {
        **thin,
        "teks": "Daftar dosen: " + "A" * 1200,
        "html": "<html><body>Daftar dosen</body></html>",
        "mesin": "b",
        "diambil_pada": "2026-11-11T09:00:00+00:00",
        "links": [
            {
                "url": "https://ftmm.unair.ac.id/prof-dr-dosen/",
                "teks": "Prof. Dosen",
                "jenis": "page",
            },
            {
                "url": "https://ftmm.unair.ac.id/wp-content/uploads/pedoman.pdf",
                "teks": "Pedoman",
                "jenis": "pdf",
            },
        ],
        "gambar": [{"url": "https://ftmm.unair.ac.id/wp-content/uploads/struktur.png", "alt": "Struktur"}],
    }
    (dump_dir / "staf-dosen.json").write_text(json.dumps(dump), encoding="utf-8")

    stats = web_crawl.merge_dump(dump_dir, out)
    rows = [json.loads(line) for line in (out / "pages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stats["diganti"] == 1
    assert len(rows) == 1
    assert rows[0]["jumlah_karakter"] > 1000
    assert rows[0]["mesin"] == "b"
    assert rows[0]["perlu_render"] is False

    edges = [json.loads(line) for line in (out / "graf.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(edge["ke"].endswith("/prof-dr-dosen/") for edge in edges)
    media = [json.loads(line) for line in (out / "media.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {item["jenis"] for item in media} == {"pdf", "image"}
