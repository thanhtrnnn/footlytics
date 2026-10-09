"""Package only public landing resources; keep design/planning files out of hosting."""
from pathlib import Path
import hashlib
import shutil
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "web"
TARGET = ROOT / "dist-web"
if TARGET.exists():
    shutil.rmtree(TARGET)
TARGET.mkdir()
for name in ("index.html", "styles.css", "site.js", "privacy.html", "404.html", "robots.txt", "sitemap.xml", "_headers"):
    shutil.copy2(SOURCE / name, TARGET / name)
shutil.copytree(SOURCE / "assets", TARGET / "assets")
shutil.copytree(SOURCE / "about", TARGET / "about")
# Changed assets get new URLs even when a returning browser retains an older copy.
for name in ("styles.css", "site.js"):
    digest = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()[:12]
    asset = Path(name)
    versioned = f"{asset.stem}.{digest}{asset.suffix}"
    shutil.copy2(SOURCE / name, TARGET / versioned)
    for page in TARGET.rglob("*.html"):
        html = page.read_text()
        page.write_text(html.replace(f'/{name}"', f'/{versioned}"'))
print(f"Cloudflare Pages package ready: {TARGET}")
