#!/usr/bin/env python3
"""
03_publish_to_wordpress.py

Watches a transcripts folder for new .txt files. If a transcript begins with a
spoken "Meta note" / "System note" block and includes:
  - "create blog post" OR "create a blog post"
then it automatically creates a WordPress DRAFT via the WP REST API.

It extracts:
  - title (spoken "Title:" or derived from body)
  - excerpt (spoken "Excerpt:" or derived)
  - tags (spoken "Tags:" or derived keywords; creates tags if missing)

Categories: intentionally NOT set (leave unassigned).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import requests

TIMESTAMP_RE = re.compile(r"^\[\d{2}:\d{2}(?::\d{2})?\]\s*")

def strip_timestamps(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        cleaned_lines.append(TIMESTAMP_RE.sub("", line))
    return "\n".join(cleaned_lines).strip()


FOOTER_BLOCK_RE = re.compile(
    r"\n---\nTranscribed locally with whisper\.cpp.*?\nChecksum:\s*[a-f0-9]{64}\s*\n?\Z",
    flags=re.S | re.I
)

def strip_footer_block(text: str) -> str:
    return FOOTER_BLOCK_RE.sub("", text).rstrip()

# ----------------------------
# Configuration defaults
# ----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRANSCRIPTS_DIR = SCRIPT_DIR / "_02_Transcripts"
DEFAULT_LOG_DIR = SCRIPT_DIR / "_LOGS"
DEFAULT_STATE_FILE = DEFAULT_LOG_DIR / "wp_publish_state.json"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "wp_publish.log"

# Only scan the top part of the transcript for meta commands
META_SCAN_MAX_LINES = 30

# Action phrases you said you might use (case-insensitive)
ACTION_PATTERNS = [
    r"\bcreate\s+blog\s+post\b",
    r"\bcreate\s+a\s+blog\s+post\b",
]

GATE_PATTERNS = [
    r"\bmeta\s+note\b",
    r"\bsystem\s+note\b",
]

# Simple stopwords for tag extraction
STOPWORDS = {
    "a","an","and","are","as","at","be","because","been","but","by",
    "can","could","did","do","does","for","from","had","has","have",
    "he","her","him","his","how","i","if","in","into","is","it","its",
    "just","like","me","more","most","my","no","not","of","on","or",
    "our","out","so","some","than","that","the","their","them","then",
    "there","these","they","this","to","up","was","we","were","what",
    "when","where","which","who","why","with","you","your"
}


@dataclass
class ParsedMeta:
    should_post: bool
    title: Optional[str]
    excerpt: Optional[str]
    tags: Optional[List[str]]
    body: str


def log(msg: str, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def load_state(state_file: Path) -> dict:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def normalize_tag(tag: str) -> str:
    t = tag.strip().lower()
    t = re.sub(r"[^a-z0-9\s\-_/]+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter (deterministic, no ML)
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def derive_title(body: str) -> str:
    # Prefer first meaningful sentence/line
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    candidate = lines[0] if lines else ""
    sents = split_sentences(candidate)
    title = sents[0] if sents else candidate

    # Clean up title length/punctuation
    title = title.strip().strip('"“”').strip()
    title = re.sub(r"\s+", " ", title)
    if len(title) > 72:
        title = title[:72].rstrip() + "…"
    # Remove trailing punctuation that looks awkward in titles
    title = re.sub(r"[.!?]+$", "", title).strip()
    return title or "Untitled Draft"


def derive_excerpt(body: str, max_chars: int = 220) -> str:
    sents = split_sentences(body)
    excerpt = " ".join(sents[:2]) if sents else body.strip()
    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip() + "…"
    return excerpt


def derive_tags(body: str, max_tags: int = 7) -> List[str]:
    text = body.lower()
    # Tokenize words
    words = re.findall(r"[a-z0-9][a-z0-9\-_/]{2,}", text)
    freq = {}
    for w in words:
        if w in STOPWORDS:
            continue
        # mild normalization
        w = w.strip("-_/")
        if len(w) < 3 or w in STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1

    # Score by frequency, then alphabetically for determinism
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    tags = [normalize_tag(k) for k, _ in ranked[:max_tags]]
    # prune empties
    tags = [t for t in tags if t]
    return tags


def parse_transcript_for_meta(text: str) -> ParsedMeta:
    lines = text.splitlines()
    head = "\n".join(lines[:META_SCAN_MAX_LINES]).lower()

    gate_ok = any(re.search(pat, head, flags=re.I) for pat in GATE_PATTERNS)
    action_ok = any(re.search(pat, head, flags=re.I) for pat in ACTION_PATTERNS)

    should_post = bool(gate_ok and action_ok)

    # Extract optional spoken fields from the head section
    title = None
    excerpt = None
    tags = None

    # Robust-ish extraction: accept "Title:" anywhere in head section
    def find_field(pattern: str) -> Optional[str]:
        m = re.search(pattern, "\n".join(lines[:META_SCAN_MAX_LINES]), flags=re.I)
        if m:
            return m.group(1).strip()
        return None

    title = find_field(r"^\s*title\s*:\s*(.+)\s*$")
    excerpt = find_field(r"^\s*excerpt\s*:\s*(.+)\s*$")
    tags_raw = find_field(r"^\s*tags\s*:\s*(.+)\s*$")
    if tags_raw:
        tags = [normalize_tag(t) for t in re.split(r"[,\|;]+", tags_raw) if normalize_tag(t)]

    # Remove the meta block from the body.
    # Strategy:
    # - If the transcript starts with meta/system note, remove lines until first blank line after any action phrase is seen.
    # - Otherwise, keep full text as body.
    body_start_idx = 0
    seen_gate_or_action = False
    for i, ln in enumerate(lines[:META_SCAN_MAX_LINES]):
        if re.search(r"(meta\s+note|system\s+note)", ln, flags=re.I) or any(re.search(p, ln, flags=re.I) for p in ACTION_PATTERNS):
            seen_gate_or_action = True
        # Once we've seen it, the first blank line ends the meta header
        if seen_gate_or_action and ln.strip() == "":
            body_start_idx = i + 1
            break

    body = "\n".join(lines[body_start_idx:]).strip()

    # If fields missing, derive from body
    if not title:
        title = derive_title(body)
    if not excerpt:
        excerpt = derive_excerpt(body)
    if not tags:
        tags = derive_tags(body)

    return ParsedMeta(
        should_post=should_post,
        title=title,
        excerpt=excerpt,
        tags=tags,
        body=body
    )


class WordPressClient:
    def __init__(self, site_url: str, username: str, app_password: str, log_file: Path):
        self.site_url = site_url.rstrip("/")
        self.api_base = f"{self.site_url}/wp-json/wp/v2"
        self.auth = (username, app_password)
        self.log_file = log_file

    def _req(self, method: str, path: str, **kwargs):
        url = f"{self.api_base}{path}"
        resp = requests.request(method, url, auth=self.auth, timeout=30, **kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(f"WP API error {resp.status_code}: {resp.text[:500]}")
        return resp

    def ensure_tag_ids(self, tag_names: List[str]) -> List[int]:
        ids: List[int] = []
        for name in tag_names:
            clean = name.strip()
            if not clean:
                continue

            # Search tags
            resp = self._req("GET", f"/tags?search={requests.utils.quote(clean)}&per_page=100")
            items = resp.json()

            tag_id = None
            # Prefer exact match (case-insensitive)
            for it in items:
                if str(it.get("name", "")).strip().lower() == clean.lower():
                    tag_id = int(it["id"])
                    break

            if tag_id is None:
                # Create tag
                resp2 = self._req("POST", "/tags", json={"name": clean})
                tag_id = int(resp2.json()["id"])
                log(f"Created new tag: '{clean}' (id={tag_id})", self.log_file)

            ids.append(tag_id)

        # Deduplicate while preserving order
        seen = set()
        out = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def create_draft_post(self, title: str, content: str, excerpt: str, tags: List[str]) -> Tuple[int, str]:
        tag_ids = self.ensure_tag_ids(tags) if tags else []
        payload = {
            "title": title,
            "content": content,
            "excerpt": excerpt,
            "status": "draft",
        }
        # Leave categories unassigned by NOT setting categories at all.
        if tag_ids:
            payload["tags"] = tag_ids

        resp = self._req("POST", "/posts", json=payload)
        data = resp.json()
        post_id = int(data["id"])
        link = data.get("link") or f"{self.site_url}/?p={post_id}"
        return post_id, link


def get_env_required(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"Missing required environment variable: {name}")
    return v


def iter_transcript_files(transcripts_dir: Path) -> List[Path]:
    return sorted([p for p in transcripts_dir.glob("*.txt") if p.is_file()])


def main():
    parser = argparse.ArgumentParser(description="Auto-publish selected transcripts to WordPress as drafts.")
    parser.add_argument("--transcripts-dir", default=str(DEFAULT_TRANSCRIPTS_DIR), help="Path to transcripts folder")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Path to logs folder")
    parser.add_argument("--poll-seconds", type=int, default=5, help="Polling interval")
    parser.add_argument("--dry-run", action="store_true", help="Parse and log, but do not post to WordPress")
    args = parser.parse_args()

    transcripts_dir = Path(args.transcripts_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_file = log_dir / "wp_publish.log"
    state_file = log_dir / "wp_publish_state.json"

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # WP credentials from env
    # Example:
    #   export WP_SITE_URL="https://yoursite.com"
    #   export WP_USERNAME="bert"
    #   export WP_APP_PASSWORD="xxxx xxxx xxxx xxxx"
    site_url = get_env_required("WP_SITE_URL")
    username = get_env_required("WP_USERNAME")
    app_password = get_env_required("WP_APP_PASSWORD")

    wp = WordPressClient(site_url, username, app_password, log_file)

    state = load_state(state_file)
    state.setdefault("posted", {})  # maps file_path -> {sha, post_id, link, posted_at}

    log(f"Watching transcripts folder: {transcripts_dir}", log_file)
    log(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'} | Poll every {args.poll_seconds}s", log_file)

    while True:
        try:
            for path in iter_transcript_files(transcripts_dir):
                key = str(path)
                raw_text = path.read_text(encoding="utf-8", errors="ignore")
                text = strip_timestamps(raw_text)
                digest = sha256_text(text)

                text = path.read_text(encoding="utf-8", errors="ignore")
                text = strip_timestamps(text)
                text = strip_footer_block(text)

                already = state["posted"].get(key)
                if already and already.get("sha") == digest:
                    continue  # unchanged and already posted/processed

                parsed = parse_transcript_for_meta(text)

                # If no intent, mark as "seen" so we don't reprocess constantly unless file changes
                if not parsed.should_post:
                    state["posted"][key] = {
                        "sha": digest,
                        "status": "ignored_no_intent",
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    save_state(state_file, state)
                    continue

                # Intent present: publish as draft
                log(f"Intent detected in: {path.name}", log_file)
                log(f"  Title: {parsed.title}", log_file)
                log(f"  Tags: {', '.join(parsed.tags or [])}", log_file)

                if args.dry_run:
                    state["posted"][key] = {
                        "sha": digest,
                        "status": "dry_run_would_post",
                        "title": parsed.title,
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    save_state(state_file, state)
                    continue

                post_id, link = wp.create_draft_post(
                    title=parsed.title or "Untitled Draft",
                    content=parsed.body,
                    excerpt=parsed.excerpt or "",
                    tags=parsed.tags or [],
                )

                log(f"Draft created: post_id={post_id} | {link}", log_file)

                state["posted"][key] = {
                    "sha": digest,
                    "status": "posted_draft",
                    "post_id": post_id,
                    "link": link,
                    "title": parsed.title,
                    "posted_at": datetime.now().isoformat(timespec="seconds"),
                }
                save_state(state_file, state)

            time.sleep(args.poll_seconds)

        except KeyboardInterrupt:
            log("Stopped by user (Ctrl+C).", log_file)
            return
        except Exception as e:
            log(f"ERROR: {e}", log_file)
            time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
