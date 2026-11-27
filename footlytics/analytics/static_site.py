"""Zero-backend static shareable web pack builder (`footlytics pack build-site`).

Generates a self-contained static directory (`index.html`, `radar/*.json`, `clips/`,
`dossier.html`, `sportscode.xml`, `playlist.csv`, `_headers`) deployable directly to
Cloudflare Pages, Netlify, S3, or a private folder so a club analyst or Head Coach can
filter sequences, watch synchronized MP4 + 2D pitch radar, mark verdicts locally, and
export filtered Sportscode XML/CSV without running Python.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..workspace_server import PLACEHOLDER, TEMPLATE, WorkspaceStore


def _sha256_passcode(passcode: str) -> str:
    return hashlib.sha256(passcode.strip().encode("utf-8")).hexdigest()


def build_static_pack_site(
    roots: Sequence[Path | str] | Path | str,
    out_dir: Path | str,
    *,
    pack_dir: Path | str | None = None,
    club: str | None = None,
    title: str | None = None,
    passcode: str | None = None,
    pre_roll_s: float = 3.0,
    post_roll_s: float = 2.0,
    radar_hz: float = 12.0,
    only_with_clips: bool = False,
) -> dict[str, Any]:
    """Build a static, zero-backend shareable Tactical Workspace site for an Opposition Pack."""
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    pdir = Path(pack_dir).resolve() if pack_dir is not None else None

    pack_dirs: list[Path] = [pdir] if pdir is not None else []
    store = WorkspaceStore(
        roots,
        pack_dirs=pack_dirs,
        pre_roll_s=pre_roll_s,
        post_roll_s=post_roll_s,
    )
    snap = store.public_snapshot()

    # Optional club filter (match opposition pack focus)
    seqs: list[dict[str, Any]] = snap["sequences"]
    if club:
        club_l = club.strip().lower()
        seqs = [s for s in seqs if s["club"].lower() == club_l]
    if only_with_clips:
        seqs = [s for s in seqs if s["has_clip"]]

    # Copy pack dossier / exports / clips into out_path when out_path != pdir
    clips_out = out_path / "clips"
    dossier_rel = ""
    if pdir is not None:
        for fname in ("dossier.html", "dossier.md", "sportscode.xml", "playlist.csv", "pack_manifest.json"):
            src_f = pdir / fname
            if src_f.is_file():
                dst_f = out_path / fname
                if src_f.resolve() != dst_f.resolve():
                    shutil.copy2(src_f, dst_f)
                if fname == "dossier.html":
                    dossier_rel = "dossier.html"
    elif (out_path / "dossier.html").is_file():
        dossier_rel = "dossier.html"

    # Pre-bake compact 2D radar JSON slices & relative clip paths
    radar_dir = out_path / "radar"
    radar_dir.mkdir(parents=True, exist_ok=True)

    radar_written = 0
    clips_linked = 0
    static_seqs: list[dict[str, Any]] = []
    for idx, item in enumerate(seqs, 1):
        rec = dict(item)
        rec["clip_index"] = idx
        rec["video_url"] = ""
        rec["has_full_video"] = False

        # Resolve relative MP4 clip path
        mp4_src = store.clip_path_for(rec["uid"])
        if mp4_src is not None and mp4_src.is_file():
            if pdir is not None and mp4_src.is_relative_to(pdir):
                rel = mp4_src.relative_to(pdir)
            else:
                rel = Path("clips") / mp4_src.parent.name / mp4_src.name
            dst_mp4 = out_path / rel
            dst_mp4.parent.mkdir(parents=True, exist_ok=True)
            if mp4_src.resolve() != dst_mp4.resolve():
                shutil.copy2(mp4_src, dst_mp4)
            rec["clip_url"] = rel.as_posix()
            rec["has_clip"] = True
            clips_linked += 1
        else:
            rec["clip_url"] = ""
            rec["has_clip"] = False

        # Export compact 2D pitch radar trajectory JSON
        if rec["has_radar"]:
            rad_name = f"seq_{idx:04d}.json"
            rad_payload = store.radar_slice(
                rec["match_id"],
                float(rec["start_s"]),
                float(rec["end_s"]),
                target_hz=radar_hz,
            )
            (radar_dir / rad_name).write_text(
                json.dumps(rad_payload, separators=(",", ":")),
                encoding="utf-8",
            )
            rec["radar_url"] = f"radar/{rad_name}"
            radar_written += 1
        else:
            rec["radar_url"] = ""

        static_seqs.append(rec)

    kept_match_ids = {s["match_id"] for s in static_seqs}
    matches_meta = [m for m in snap["matches"] if m["match_id"] in kept_match_ids] or snap["matches"]
    clubs = sorted({s["club"] for s in static_seqs if s["club"]})
    labels = [lbl for lbl in snap["labels"] if any(s["label"] == lbl for s in static_seqs)]

    site_title = title or (f"{club.upper()} Opposition Pack" if club else "Footlytics Opposition Pack")
    clean_pass = passcode.strip() if passcode and passcode.strip() else None

    static_snap: dict[str, Any] = {
        "static_mode": True,
        "pack_title": site_title,
        "dossier_url": dossier_rel,
        "passcode_sha256": _sha256_passcode(clean_pass) if clean_pass else "",
        "reviewer": snap["reviewer"],
        "matches": matches_meta,
        "clubs": clubs,
        "labels": labels,
        "label_vi": snap["label_vi"],
        "sequences": static_seqs,
        "profile": snap.get("profile"),
        "presets": snap.get("presets", []),
    }

    blob = json.dumps(static_snap, separators=(",", ":")).replace("</", "<\\/")
    html_text = TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, blob)
    (out_path / "index.html").write_text(html_text, encoding="utf-8")

    # Cloudflare Pages / static host headers: prevent search-engine indexing of club packs
    headers_text = (
        "/*\n"
        "  X-Robots-Tag: noindex, nofollow\n"
        "  X-Content-Type-Options: nosniff\n"
        "  Referrer-Policy: no-referrer\n"
        "\n"
        "/clips/*\n"
        "  Cache-Control: public, max-age=604800\n"
        "\n"
        "/radar/*\n"
        "  Cache-Control: public, max-age=86400\n"
    )
    (out_path / "_headers").write_text(headers_text, encoding="utf-8")
    (out_path / "robots.txt").write_text("User-agent: *\nDisallow: /\n", encoding="utf-8")

    manifest = {
        "title": site_title,
        "out_dir": str(out_path),
        "index_html": str(out_path / "index.html"),
        "dossier_html": str(out_path / dossier_rel) if dossier_rel else "",
        "sequences_count": len(static_seqs),
        "clips_count": clips_linked,
        "radar_files_count": radar_written,
        "passcode_gate": "client-side, not access control" if clean_pass else "none",
        "access_control": "none: every file in this folder is publicly fetchable by anyone with its URL",
        "clips_dir": str(clips_out) if clips_out.exists() else "",
    }
    (out_path / "site_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
