# Playbook scraping ftmm.unair.ac.id

Dokumen ini mencatat cara yang terbukti bekerja pada 2026-11-11, termasuk perintah, potongan kode,
jebakan situs, dan skema data. Tujuannya: agent berikutnya bisa melanjutkan tanpa mengulang percobaan.

Lokasi data: `web_kb/` (gitignored, ±161 MB saat run pertama). Skrip dan parser: `lib/`, `scripts/`.

## 1. Temuan inti

| Fakta | Bukti |
|---|---|
| Situs WordPress + Elementor/ElementsKit, LiteSpeed Cache, Yoast SEO, EventOn, GTranslate, DataTables | ekspor Wappalyzer pengguna |
| REST API publik: `/wp-json/wp/v2/{pages,posts,ajde_events}` | 200, JSON |
| Client HTTP biasa (`urllib`, `curl`) diblokir WAF | 200 + body 242 byte "The requested URL was rejected" |
| Chromium sungguhan lolos | 200, JSON, UA `Chrome/150.0.0.0` |
| Blokir bersifat per-client, bukan sekadar IP | HTTP tetap ditolak setelah 9 jam; browser tetap 200 |

Konsekuensi: **selalu pakai browser** untuk situs ini. Jangan mengganti UA atau memakai TLS
impersonation; yang berhasil adalah browser asli, bukan header palsu.

## 2. Artefak

```
web_kb/
├── pages.jsonl          # satu baris = satu dokumen
├── urls.jsonl           # URL dari sitemap Yoast setelah filter
├── graf.jsonl           # edge: dari, ke, anchor, jenis
├── media.jsonl          # PDF/gambar + halaman sumber
├── render_queue.jsonl   # halaman tipis yang perlu render
├── manifest.json        # provenance + hash konten + statistik
├── log.jsonl            # percobaan HTTP (umumnya blokir)
├── log_browser.jsonl    # jejak per respons browser (REST, sitemap, render)
├── raw_browser_rest/    # wrapper REST + XML sitemap mentah
├── raw_browser/         # dump render: teks, html, links, gambar, ua, viewport
├── media/               # berkas biner (JPG struktur, PDF pedoman)
├── ocr/                 # hasil tesseract
├── pdf/                 # hasil pdftotext
├── media_lokal.json     # peta berkas lokal -> URL sumber -> path teks
└── index/ ... (tidak dipakai; index Dense tetap di ./index/)
```

Skema baris `pages.jsonl`:

```json
{
  "id": 18929,
  "tipe": "page|post|event|dokumen",
  "url": "https://ftmm.unair.ac.id/...",
  "slug": "...",
  "judul": "...",
  "tanggal": "", "modified": "",
  "prodi": "Teknik Elektro|Teknologi Sains Data|...|",
  "teks": "...",
  "jumlah_html": 0,
  "jumlah_karakter": 619,
  "hash_teks": "sha256",
  "diambil_pada": "ISO-8601",
  "sumber_raw": "pages-0001.json",
  "links": [{"url": "...", "teks": "...", "jenis": "page|pdf|image|language|external|skip"}],
  "gambar": [{"url": "...", "alt": "..."}],
  "status": "ok|kosong",
  "perlu_render": false,
  "mesin": "b"
}
```

### 2.1 Artefak kategori (turunan)

`scripts/categorize_web_kb.py` membaca `web_kb/pages.jsonl`, membuang URL duplikat
(ambil teks terpanjang), menormalkan judul/teks, lalu menulis satu JSON per kategori
ke `web_kb_categorized/` (gitignored):

```bash
uv run python scripts/categorize_web_kb.py
```

Setiap dokumen berisi `id, tipe, kategori, dasar_kategori, judul, url, slug, tanggal,
modified, prodi, jumlah_karakter, hash_teks, teks`. `manifest.json` mencatat jumlah per
kategori, jumlah baris yang dibuang, dan aturan klasifikasi. Kategori: `akademik`,
`program_studi`, `profil_staf`, `profil_fakultas`, `kemahasiswaan`, `prestasi`,
`penelitian`, `pengabdian`, `kerjasama`, `fasilitas`, `regulasi`, `mutu`, `agenda`,
`sdgs`, `dokumen_resmi`, `faq`, `berita`, `lainnya`.

## 3. Alur yang terbukti

### 3.1 Buka tab dan probe

```js
const tab = await browser.open({
  name: "ftmm-crawl",
  url: "https://ftmm.unair.ac.id/",
  allowed_domains: ["ftmm.unair.ac.id"],
  wait_until: "domcontentloaded",
  timeout: 60,
  persist: true,
});
```

`allowed_domains` membatasi seluruh request (termasuk analytics) ke domain situs.
`persist: true` agar tab hidup antar turn.

### 3.2 Ambil JSON dari konteks halaman

```js
async function ambil(path) {
  const res = await tab.run(async ({ page }, p) => {
    return await page.evaluate(async (u) => {
      const r = await fetch(u, { credentials: "same-origin" });
      return {
        status: r.status,
        body: await r.text(),
        total: r.headers.get("x-wp-total"),
      };
    }, p);
  }, { args: [path] });
  if (res.status !== 200 || res.body.includes("requested URL was rejected")) {
    throw new Error(`blokir ${path} status=${res.status}`);
  }
  return res;
}
```

Pagination stabil: `orderby=id&order=asc&per_page=100&page=N&_fields=id,slug,link,date,modified,title,content,excerpt`.
Berhenti saat `items.length < 100` atau HTTP 400 `rest_post_invalid_page_number`.

### 3.3 Simpan wrapper + resume

```js
const DIR = "<repo>/web_kb/raw_browser_rest";
const wrapper = { url, status: 200, byte: body.length, diambil_pada: new Date().toISOString(),
                  total_situs: total ? Number(total) : null, items: JSON.parse(body) };
await Bun.write(`${DIR}/${base}-${String(page).padStart(4, "0")}.json`, JSON.stringify(wrapper));
await new Promise((r) => setTimeout(r, 6000)); // jeda antar request
```

Jenis yang diambil: semua `rest_base` dari `/wp-json/wp/v2/types` kecuali template/widget:
`attachment, nav_menu_item, wp_block, wp_template, wp_template_part, wp_navigation,
wp_font_family, wp_font_face, wp_global_styles, elementor_library, elementor_snippet,
elementskit_content, elementskit_template, e-floating-buttons, custom_css, customize_changeset,
oembed_cache, user_request`.

### 3.4 Susun korpus

```bash
uv run python scripts/import_browser_rest.py   # raw_browser_rest/ -> pages.jsonl, graf.jsonl, urls.jsonl
uv run python scripts/import_browser_dump.py   # raw_browser/ -> replace teks halaman tipis
```

`import_browser_rest.py` juga menulis `log_browser.jsonl` dan menambahkan dokumen dari
`media_lokal.json` (OCR/PDF) sebagai `tipe="dokumen"`.

### 3.5 Render halaman tipis

`render_queue.jsonl` dihasilkan dari `perlu_render` (teks < 400 karakter atau memuat penanda
`dataTable`/`admin-ajax.php`). Render dengan ekstraksi DOM:

```js
const EXTRACT = `(() => {
  const buang = "script,style,noscript,svg,iframe,header,footer,nav,.elementor-location-header,.elementor-location-footer,[class*=accessibility],[class*=gtranslate]";
  const pilih = [".elementor-location-single", "main", "article", ".site-main", "#content"];
  let root = null;
  for (const s of pilih) { const el = document.querySelector(s); if (el && (el.innerText || "").trim().length > 40) { root = el; break; } }
  root = root || document.body;
  root.querySelectorAll(buang).forEach((el) => el.remove());
  return {
    judul: document.title,
    teks: (root.innerText || "").replace(/\\n{3,}/g, "\\n\\n").trim(),
    html: (root.innerHTML || "").slice(0, 300000),
    links: [...root.querySelectorAll("a[href]")].slice(0, 600).map((a) => ({ url: a.href, teks: (a.innerText || "").trim() })),
    gambar: [...root.querySelectorAll("img")].slice(0, 200).map((i) => ({ url: i.currentSrc || i.src, alt: i.alt || "" })),
    ua: navigator.userAgent, viewport: window.innerWidth + "x" + window.innerHeight
  };
})()`;

await tab.goto(url);
await new Promise((r) => setTimeout(r, 1500));
const data = await tab.evaluate(EXTRACT);
await Bun.write(`${DIR}/raw_browser/${Bun.hash(url).toString(16)}.json`,
  JSON.stringify({ ...data, url, mesin: "b", diambil_pada: new Date().toISOString() }));
```

Nama berkas memakai `Bun.hash(url)` supaya resume hanya memeriksa `Bun.file(path).exists()`.
Jangan pakai `tab.waitFor(1500)`; helper itu menerima selector string, bukan angka.
Ekstraksi harus lewat `tab.evaluate`, bukan `page.evaluate` di dalam `tab.run` dengan closure
(karena closure sel tidak terbawa).

### 3.6 Unduh biner dan ekstrak teks

```js
const res = await tab.evaluate(`(async () => {
  const r = await fetch(${JSON.stringify(url)}, { credentials: "same-origin" });
  const b = new Uint8Array(await r.arrayBuffer());
  let s = ""; const step = 32768;
  for (let i = 0; i < b.length; i += step) s += String.fromCharCode.apply(null, b.subarray(i, i + step));
  return { status: r.status, ctype: r.headers.get("content-type"), b64: btoa(s), byte: b.length };
})()`);
await Bun.write(path, Buffer.from(res.b64, "base64"));
```

```bash
tesseract web_kb/media/struktur-organisasi.jpg web_kb/ocr/struktur-organisasi -l ind+eng
pdftotext -layout web_kb/media/<nama>.pdf web_kb/pdf/<nama>.txt
# PDF hasil scan: render dulu per halaman
pdftoppm -r 200 -png web_kb/media/<nama>.pdf /tmp/hal
for f in /tmp/hal-*.png; do tesseract "$f" "${f%.png}" -l ind+eng; done
```

Catat setiap berkas di `web_kb/media_lokal.json` (`nama -> {url, judul, teks}`) agar ikut masuk
korpus saat impor ulang.

## 4. Aturan aman (jangan dilanggar)

1. Satu tab, satu request, jeda 5–6 detik. Tanpa paralel.
2. Tanpa override UA/header, tanpa `curl_cffi`, tanpa rotasi IP/proxy.
3. `allowed_domains` dibatasi ke `ftmm.unair.ac.id`.
4. Deteksi penolakan dari body, bukan status: `"requested url was rejected"` atau `"request rejected"`,
   atau body HTML ≤ 300 byte padahal endpoint seharusnya JSON.
5. Saat diblokir: berhenti. Jangan retry cepat; tunggu ≥30 menit. Satu probe per percobaan.
6. `robots.txt` situs mengizinkan semua path; tetap jaga laju.
7. `web_kb/` tidak boleh di-commit; berisi teks institusi dan data pribadi dosen.

## 5. Jebakan spesifik situs

| Gejala | Sebab | Tindakan |
|---|---|---|
| HTTP 200 tapi body kecil | WAF F5 mengembalikan halaman penolakan | deteksi body, stop |
| `post-sitemap2.xml`/`3.xml` berisi "Request Rejected" | path diblokir WAF | skip, catat di manifest; post tetap lengkap via REST |
| EventOn `content.rendered` kosong | deskripsi event disimpan di meta, bukan post content | render `/events/<slug>/` |
| Halaman `< 400` karakter | konten dimuat JS/tabel | render browser |
| Link `/en/...` | GTranslate | jangan diikuti; ambil versi Indonesia |
| HTML dari cache | LiteSpeed Cache | catat header `x-litespeed-cache` di log; jangan cache-bust |
| Isi accordion/tab "hilang" | Elementor menyembunyikan via CSS | ambil `innerText`/HTML tanpa filter visibilitas |
| Profil dosen tanpa prodi | halaman induk yang menandai prodi | `propagate_prodi()` mewarisi dari halaman staf |
| `3d-flip-book/*` tidak ada di REST | CPT tidak terekspos | tercatat sebagai gap; render bila perlu |

## 6. Verifikasi

```bash
uv run pytest -q                                   # TEST DONE
uv run python scripts/import_browser_rest.py       # [OK] N dokumen ...
uv run python scripts/import_browser_dump.py       # [OK] N dokumen (ditambah=..., diganti=...)
uv run python scripts/scrape_ftmm.py --dry-run --delay 8   # hanya untuk jalur HTTP; akan [FAIL] bila WAF
```

Cek cepat data:

```bash
python3 -c "import json;rows=[json.loads(l) for l in open('web_kb/pages.jsonl')];print(len(rows),sum(r['status']=='ok' for r in rows))"
python3 -c "import json;m=json.load(open('web_kb/manifest.json'));print(m['jumlah_dokumen'],m['jumlah_kosong'],m['graf'])"
```

## 7. Data yang berhasil (run 2026-11-11)

Angka final selalu dibaca dari `manifest.json`; tabel ini hanya contoh isi.

| Kategori | Jumlah | Contoh |
|---|---|---|
| Page | 184 | `tentang-fakultas/` (3.332 char), `sejarah-fakultas-2/` (5.850), `pimpinan-fakultas/` (767) |
| Post/berita | 3.331 | `mengapa-belajar-robotika-dan-kecerdasan-buatan/` (1.893), `kompetensi-lulusan-teknik-robotika-...` |
| Event | 191 | `/events/forum-rektor-indonesia-2022/` (206), `/events/dies-natalis-ftmm/` |
| Dokumen OCR/PDF | 9 | Struktur Organisasi (1.186), Buku Pedoman 2020/2021 (1.129.140) |
| Profil pimpinan | 4 | `/retna-apsari/`, `/dr-nimatuzahroh/`, `/dr-fadli-ama/`, `/prastika-krisma-jiwanti-.../` |
| Profil dosen | 52+ | `/prisma-megantoro-s-t-m-eng/` (622, prodi TE), `/indah-fahmiyah-s-si-m-stat/` (291, TSD) |
| Portal informasi | 86 | `/portal-informasi/cuti-akademik/`, `/skma/`, `/transkrip-nilai/` |

Trace: `pimpinan-fakultas` → 4 edge profil; `staf-dosen-prodi-*` → 52 edge profil; total 13.330 edge.

## 8. Integrasi retrieval

```bash
uv run python build_index.py --web          # butuh Ollama hidup
uv run python run_bm25.py --web             # tanpa Ollama
FTMM_WEB=1 uv run streamlit run app.py
```

`load_web_pages()` membaca `pages.jsonl`, melewati `status != ok`, dan memberi `prodi` hasil
inferensi/warisan. `load_documents()` tetap default tanpa web agar `corpus_hash` eval lama tidak berubah.

## 9. Gap yang masih terbuka

- 3 post flipbook kosong (`booklet-ftmm`, `buku-aturan-berperilaku`, `pedoman-pendidikan` versi post).
- 216 URL sitemap tidak diambil, mayoritas `/3d-flip-book/`.
- `post-sitemap2.xml`, `post-sitemap3.xml` diblokir WAF.
- 35 baris sisa `render_queue.jsonl` (sebagian duplikat; 1 URL `download-center/` ditolak).
- Index Dense belum dibuat saat run (Ollama tidak aktif).

## 10. Riwayat

| Commit | Isi |
|---|---|
| `7e1d456` | scraper HTTP awal, deteksi WAF, isolasi `web_kb/` |
| `29ad303` | crawl via browser, sitemap Yoast, render halaman tipis, OCR/PDF, importer |
