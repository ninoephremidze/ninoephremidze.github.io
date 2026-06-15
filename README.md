# Academic website

A clean, single-file academic site. Plain HTML and CSS, no build step, no
dependencies — it runs anywhere static files do, including GitHub Pages.

## Deploy to GitHub Pages

You have two options. The first gives you the nicest URL.

### Option A — user site (`yourname.github.io`)
1. Create a new public repo named **exactly** `yourusername.github.io`
   (use your real GitHub username).
2. Add `index.html` (and `cv.pdf`, `photo.jpg` once you have them).
3. Push. Your site is live at `https://yourusername.github.io` within a minute
   or two. No settings to change — this repo name is special.

### Option B — project site (any repo name)
1. Put `index.html` in any repo (e.g. `website`).
2. Repo **Settings → Pages → Build and deployment → Source: Deploy from a
   branch**, branch `main`, folder `/ (root)`. Save.
3. Live at `https://yourusername.github.io/website/`.

### Custom domain (optional, ~$12/yr)
Buy `yourname.com`, then in **Settings → Pages → Custom domain** enter it and
follow the DNS instructions. Worth it — the URL stays yours across institutions.

## Customize

Open `index.html` and search for `EDIT:` — every spot you need to change is
marked. The essentials:

- **Name, role, affiliation** — top of the sidebar.
- **Links** — replace each `#` with your real Google Scholar / ORCID / GitHub
  URLs; delete any line you don't use.
- **Photo** — add `photo.jpg` to the repo, then swap the monogram block for a
  photo as described in the CSS comment near `.avatar` (the `<div class="avatar">`
  becomes `<img class="avatar" src="photo.jpg" alt="Your Name">`).
- **CV** — add your `cv.pdf` to the repo; the Download CV button already points
  to it.
- **Publications** — each paper is one `<article class="pub">` block. Copy it,
  edit, and wrap your own name in `<span class="me">…</span>` so it stands out.
- **News** — keep it current, or delete the whole `#news` section if you won't
  update it. A stale news feed looks worse than none.

## Make it yours

The accent colour, fonts, and spacing are all CSS variables at the very top of
the file (`:root { … }`). Change `--accent` to recolour every link, button, and
marker at once. Fonts load from Google Fonts via the `<link>` in the `<head>`.

## When to outgrow this

This single page covers what a PhD student needs. If you later want a blog, many
project pages, or automatic publication lists from a `.bib` file, look at the
**academicpages** or **al-folio** Jekyll templates — both run on GitHub Pages and
you can carry this content straight over.
