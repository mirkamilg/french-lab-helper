"""French Lab helper: a small MCP server the French Lab artifacts reach as a claude.ai connector.

It does the work an artifact cannot do from inside its sandbox: fetching subtitles from
YouTube and Bilibili, and talking to the 法语助手 (Eudic) OpenAPI. Secrets live in the
environment (Render environment variables), never in the artifacts.

The MCP endpoint is served at /mcp/<CONNECTOR_KEY>/ so that only someone holding the
connector URL can call it.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import re
import threading
import time
import urllib.parse
from functools import reduce

import requests
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

VERSION = "0.9.0"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36")
KEY = os.environ.get("CONNECTOR_KEY", "dev")
EUDIC_BASE = "https://api.frdic.com/api/open/v1"
READ = ToolAnnotations(readOnlyHint=True, openWorldHint=True)


def env(name: str) -> str:
    """A secret's value; one replaced later with `deploy_helper.py set-key` (stored as NAME__NEW) wins."""
    return (os.environ.get(name + "__NEW") or os.environ.get(name) or "").strip()

mcp = FastMCP(
    "French Lab",
    instructions=("Fetches French subtitles from YouTube and Bilibili, syncs with 法语助手 (Eudic), and keeps the user's "
                  "Réflexes notebook (their phone app for everyday spoken French). In any chat: when the user shares French they "
                  "heard, asks how to say something, tries out a sentence or asks you to keep something, answer them as usual and "
                  "also save it with reflexes_save, without asking. When they ask to practise or be quizzed, use reflexes_due "
                  "and report outcomes with reflexes_result."),
    host="0.0.0.0",
    port=int(os.environ.get("PORT", "7860")),
    streamable_http_path=f"/mcp/{KEY}",
    stateless_http=True,
    json_response=True,
)


# ---------------------------------------------------------------------------
# Subtitle parsing
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("​", "")
    text = re.sub(r"(?<=[A-Za-zÀ-ÿ]) ['’](?=[A-Za-zÀ-ÿ])", "'", text)
    return re.sub(r"\s+", " ", text).strip()


def _line(start: float, end: float, text: str) -> dict | None:
    text = _clean(text)
    if not text or re.fullmatch(r"\[[^\]]*\]", text):  # drop empty and [Musique]-style cues
        return None
    return {"t": round(start, 2), "d": round(max(end - start, 0), 2), "text": text}


def _ts(s: str) -> float:
    s = s.strip().replace(",", ".")
    parts = [float(p) for p in s.split(":")]
    return reduce(lambda a, b: a * 60 + b, parts)


def parse_subtitles(raw: str, fmt: str) -> list[dict]:
    """Parse json3 / srt / vtt / Bilibili JSON into [{t, d, text}]."""
    out: list[dict] = []
    if fmt == "json3":
        for ev in json.loads(raw).get("events", []):
            segs = ev.get("segs")
            if not segs:
                continue
            start = ev.get("tStartMs", 0) / 1000
            ln = _line(start, start + ev.get("dDurationMs", 0) / 1000, "".join(s.get("utf8", "") for s in segs))
            if ln:
                out.append(ln)
    elif fmt == "bili":
        for c in json.loads(raw).get("body", []):
            ln = _line(c.get("from", 0), c.get("to", 0), c.get("content", ""))
            if ln:
                out.append(ln)
    else:  # srt / vtt
        for block in re.split(r"\n\s*\n", raw.replace("\r", "")):
            m = re.search(r"([\d:.,]+)\s*-->\s*([\d:.,]+)[^\n]*\n(.*)", block, re.S)
            if m:
                ln = _line(_ts(m.group(1)), _ts(m.group(2)), m.group(3).replace("\n", " "))
                if ln:
                    out.append(ln)
    # YouTube auto captions repeat rolling text; drop exact consecutive repeats.
    dedup: list[dict] = []
    for ln in out:
        if dedup and dedup[-1]["text"] == ln["text"]:
            dedup[-1]["d"] = round(ln["t"] + ln["d"] - dedup[-1]["t"], 2)
            continue
        dedup.append(ln)
    return dedup


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

YT_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/live/|/embed/)([A-Za-z0-9_-]{11})")


def youtube_id(url: str) -> str | None:
    m = YT_ID.search(url)
    if m:
        return m.group(1)
    return url if re.fullmatch(r"[A-Za-z0-9_-]{11}", url) else None


def youtube_meta(vid: str) -> dict:
    try:
        r = requests.get("https://www.youtube.com/oembed",
                         params={"url": f"https://www.youtube.com/watch?v={vid}", "format": "json"},
                         headers={"User-Agent": UA}, timeout=15)
        if r.ok:
            j = r.json()
            return {"title": j.get("title"), "channel": j.get("author_name"),
                    "thumbnail": j.get("thumbnail_url")}
    except requests.RequestException:
        pass
    return {}


FR_WORDS = set("le la les de des du et est un une que qui pas pour dans ce il elle on vous nous je à au aux en ne se sur "
               "avec mais ou plus c'est ça cette ces son sa ses leur être avoir fait très aussi comme tout".split())


def looks_french(lines: list[dict]) -> bool:
    """Is this caption text French, whatever language the track claims? (ARTE tags French videos as German.)"""
    ws = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", " ".join(l["text"] for l in lines[:150]).lower())
    return len(ws) >= 20 and sum(w in FR_WORDS for w in ws) / len(ws) > 0.15


def youtube_transcript(vid: str, attempts: list) -> tuple[list[dict], dict] | None:
    # 1) youtube-transcript-api (innertube); 2) yt-dlp; 3) Supadata. Each failure is recorded.
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        tl = YouTubeTranscriptApi().list(vid)
        tracks = sorted(tl, key=lambda t: (not t.language_code.lower().startswith("fr"), t.is_generated))
        for t in tracks[:4]:
            lines = [ln for s in t.fetch() if (ln := _line(s.start, s.start + s.duration, s.text))]
            if t.language_code.lower().startswith("fr") or looks_french(lines):
                return lines, {"lang": t.language_code, "kind": "auto" if t.is_generated else "manual",
                               "method": "transcript-api"}
        attempts.append({"method": "transcript-api", "error": "no French track",
                         "available": [t.language_code for t in tl]})
    except Exception as e:  # noqa: BLE001 - report every failure mode to the caller
        attempts.append({"method": "transcript-api", "error": f"{type(e).__name__}: {str(e)[:300]}"})

    got = ytdlp_subtitles(f"https://www.youtube.com/watch?v={vid}", attempts)
    return got or supadata_transcript(vid, attempts)


def _supadata_get(params: dict, max_wait: int) -> dict:
    key = env("SUPADATA_API_KEY")
    if not key:
        raise ValueError("SUPADATA_API_KEY is not set")
    r = requests.get("https://api.supadata.ai/v1/transcript", headers={"x-api-key": key}, timeout=90, params=params)
    if r.status_code == 202:  # long or generated transcript: poll the job
        job = json.loads(r.content.decode("utf-8")).get("jobId")
        t0 = time.time()
        while time.time() - t0 < max_wait:
            time.sleep(3)
            r = requests.get(f"https://api.supadata.ai/v1/transcript/{job}", headers={"x-api-key": key}, timeout=30)
            if r.ok and json.loads(r.content.decode("utf-8")).get("status") not in ("queued", "active"):
                break
    if r.status_code in (402, 429):
        raise ValueError(f"QUOTA: Supadata credits are used up for this month (HTTP {r.status_code}).")
    if r.status_code >= 400:
        raise ValueError(f"Supadata HTTP {r.status_code}: {r.content.decode('utf-8', 'replace')[:300]}")
    j = json.loads(r.content.decode("utf-8"))
    if j.get("status") == "failed":
        raise ValueError(f"Supadata could not transcribe it: {str(j.get('error'))[:300]}")
    return j



def resegment(chunks: list[dict], max_words: int = 22) -> list[dict]:
    """Turn fixed-size AI transcript chunks into sentence lines.

    Supadata's generated transcripts come in ~110-character pieces that cut words in half, re-transcribe
    a few words where its audio windows overlap ("…sécurité à long terme il existe rapide ou la sécurité à
    long terme. Il existe…") and space hyphens ("c'est -à -dire"). Words get times interpolated inside their
    chunk; the earlier copy of a repeated run of ≥4 words (within 20 words) is dropped.
    """
    words = []  # [text, time]
    for c in chunks:
        text = re.sub(r"\s+-\s*(?=\w)", "-", c["text"])
        text = re.sub(r"(?<=\w)\s+'(?=\w)", "'", text)
        toks = text.split()
        if not toks:
            continue
        total = sum(len(t) + 1 for t in toks)
        pos = 0
        for t in toks:
            words.append([t, c["t"] + c["d"] * pos / max(total, 1)])
            pos += len(t) + 1
    # glue pieces split mid-word across chunk borders: "qu" + "'on" -> "qu'on"
    glued = []
    for w in words:
        if glued and (w[0].startswith("'") or glued[-1][0].endswith("'")):
            glued[-1][0] += w[0]
        else:
            glued.append(w)
    words = glued
    key = lambda w: re.sub(r"[^\w']", "", w[0].lower())
    i = 0
    out = []
    while i < len(words):
        cut = None
        for n in (8, 7, 6, 5, 4):
            if i + n > len(words):
                continue
            seq = [key(w) for w in words[i:i + n]]
            for j in range(i + 1, min(i + 21, len(words) - n + 1)):
                if [key(w) for w in words[j:j + n]] == seq:
                    cut = j
                    break
            if cut:
                break
        if cut:
            i = cut  # drop the garbled first copy, keep the clean second one
            continue
        out.append(words[i])
        i += 1
    lines, cur = [], []
    for k, w in enumerate(out):
        cur.append(w)
        nxt = out[k + 1][1] if k + 1 < len(out) else w[1] + 1.5
        end_sentence = re.search(r"[.?!…]$", w[0]) is not None
        if end_sentence or len(cur) >= max_words or (len(cur) >= 12 and w[0].endswith(",")):
            lines.append((cur, nxt))
            cur = []
    if cur:
        lines.append((cur, cur[-1][1] + 1.5))
    res = []
    starts_sentence = True
    for ws, end in lines:
        text = " ".join(w[0] for w in ws)
        if text and starts_sentence:
            text = text[0].upper() + text[1:]
        starts_sentence = bool(re.search(r"[.?!…]$", text))
        ln = _line(ws[0][1], max(end, ws[0][1] + 0.5), text)
        if ln:
            res.append(ln)
    return res


def _supadata_lines(j: dict) -> tuple[list[dict], str | None]:
    content = j.get("content") or (j.get("result") or {}).get("content") or []
    lang = j.get("lang") or (j.get("result") or {}).get("lang")
    lines = [ln for c in content if (ln := _line(c["offset"] / 1000, (c["offset"] + c.get("duration", 0)) / 1000, c["text"]))]
    return lines, lang


def supadata_transcript(vid: str, attempts: list) -> tuple[list[dict], dict] | None:
    """Transcript service fallback for cloud IPs that YouTube blocks (1 credit per native transcript)."""
    try:
        j = _supadata_get({"url": f"https://www.youtube.com/watch?v={vid}", "lang": "fr", "mode": "native"}, 80)
        lines, lang = _supadata_lines(j)
        if not str(lang or "").startswith("fr") and not looks_french(lines):
            attempts.append({"method": "supadata", "error": f"no French transcript (got {lang})",
                             "available": j.get("availableLangs")})
            return None
        return lines, {"lang": lang, "kind": "youtube", "method": "supadata"}
    except Exception as e:  # noqa: BLE001
        attempts.append({"method": "supadata", "error": f"{type(e).__name__}: {str(e)[:300]}"})
        return None


@mcp.tool(annotations=READ, structured_output=False)
def supadata_status() -> dict:
    """Supadata plan and credits for this month (AI transcripts of YouTube videos cost ~2 credits per minute)."""
    key = env("SUPADATA_API_KEY")
    if not key:
        return {"configured": False}
    r = requests.get("https://api.supadata.ai/v1/me", headers={"x-api-key": key}, timeout=20)
    if r.status_code >= 400:
        raise ValueError(f"Supadata HTTP {r.status_code}: {r.content.decode('utf-8', 'replace')[:200]}")
    j = json.loads(r.content.decode("utf-8"))
    used, mx = int(j.get("usedCredits") or 0), int(j.get("maxCredits") or 0)
    return {"configured": True, "plan": j.get("plan"), "used": used, "max": mx, "left": max(0, mx - used)}


def caption_warning(meta: dict) -> str | None:
    lang, kind = str(meta.get("lang") or ""), meta.get("kind")
    if lang and not lang.startswith("fr"):
        return (f"YouTube files this video's captions as « {lang} », so the French was recognised as the wrong "
                "language and the captions are full of errors. An AI transcript from the audio will be much better.")
    if kind in ("auto", "youtube"):
        return "These are YouTube's automatic captions: expect some recognition errors."
    return None


def ytdlp_subtitles(url: str, attempts: list, cookies: dict | None = None) -> tuple[list[dict], dict] | None:
    try:
        import yt_dlp
        opts = {"skip_download": True, "quiet": True, "no_warnings": True, "ignore_no_formats_error": True,
                "extractor_args": {"youtube": {"player_client": ["android_vr", "default"]}}}
        with yt_dlp.YoutubeDL(opts) as ydl:
            if cookies:
                for k, v in cookies.items():
                    ydl.cookiejar.set_cookie(_cookie(k, v, ".bilibili.com"))
            info = ydl.extract_info(url, download=False)
            for kind, table in (("manual", info.get("subtitles") or {}),
                                ("auto", info.get("automatic_captions") or {})):
                langs = [l for l in table if l.lower().startswith(("fr", "ai-fr"))]
                for lang in sorted(langs, key=lambda l: (l != "fr", "orig" not in l)):
                    by_ext = {f.get("ext"): f for f in table[lang]}
                    for ext in ("json3", "srt", "vtt", "json"):
                        f = by_ext.get(ext)
                        if not f:
                            continue
                        raw = f.get("data") or ydl.urlopen(f["url"]).read().decode("utf-8", "replace")
                        fmt = "bili" if ext == "json" else ext
                        lines = parse_subtitles(raw, fmt)
                        if lines:
                            meta = {"lang": lang, "kind": kind, "method": "yt-dlp",
                                    "title": info.get("title"), "channel": info.get("uploader"),
                                    "duration": info.get("duration"), "thumbnail": info.get("thumbnail")}
                            return lines, meta
            attempts.append({"method": "yt-dlp", "error": "no French subtitles",
                             "available": sorted(set(info.get("subtitles") or {}) |
                                                 set((info.get("automatic_captions") or {}).keys()))[:40]})
    except Exception as e:  # noqa: BLE001
        attempts.append({"method": "yt-dlp", "error": f"{type(e).__name__}: {str(e)[:300]}"})
    return None


def _cookie(name: str, value: str, domain: str):
    from http.cookiejar import Cookie
    return Cookie(0, name, value, None, False, domain, True, True, "/", True, True,
                  int(time.time()) + 86400 * 30, False, None, None, {})


# ---------------------------------------------------------------------------
# Bilibili
# ---------------------------------------------------------------------------

MIXIN = [46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49, 33, 9, 42,
         19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51,
         30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52]


def bili_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Referer": "https://www.bilibili.com/",
                      "Origin": "https://www.bilibili.com"})
    sess = env("BILI_SESSDATA")
    if sess:
        s.cookies.set("SESSDATA", sess, domain=".bilibili.com")
    try:
        spi = s.get("https://api.bilibili.com/x/frontend/finger/spi", timeout=10).json()["data"]
        s.cookies.set("buvid3", spi["b_3"], domain=".bilibili.com")
        s.cookies.set("buvid4", spi["b_4"], domain=".bilibili.com")
    except Exception:  # noqa: BLE001 - buvid only lowers the risk-control rate
        pass
    return s


def bili_sign(s: requests.Session, params: dict) -> dict:
    nav = s.get("https://api.bilibili.com/x/web-interface/nav", timeout=10).json()["data"]["wbi_img"]
    img = nav["img_url"].rsplit("/", 1)[1].split(".")[0]
    sub = nav["sub_url"].rsplit("/", 1)[1].split(".")[0]
    orig = img + sub
    mixin = "".join(orig[i] for i in MIXIN)[:32]
    p = dict(params, wts=int(time.time()))
    p = {k: re.sub(r"[!'()*]", "", str(v)) for k, v in sorted(p.items())}
    q = urllib.parse.urlencode(p)
    p["w_rid"] = hashlib.md5((q + mixin).encode()).hexdigest()
    return p


def bili_ids(url: str) -> tuple[dict, int]:
    if "b23.tv" in url:
        url = requests.get(url, headers={"User-Agent": UA}, timeout=10, allow_redirects=True).url
    page = int((urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("p") or ["1"])[0])
    m = re.search(r"(BV[0-9A-Za-z]{10})", url)
    if m:
        return {"bvid": m.group(1)}, page
    m = re.search(r"av(\d+)", url, re.I)
    if m:
        return {"aid": m.group(1)}, page
    raise ValueError("Not a Bilibili video link")


def bili_video(s: requests.Session, url: str) -> tuple[dict, dict]:
    """Resolve a Bilibili link to (meta, page) through the signed view API."""
    ident, page = bili_ids(url)
    view = s.get("https://api.bilibili.com/x/web-interface/wbi/view", params=bili_sign(s, ident), timeout=15).json()
    if view.get("code") != 0:
        raise ValueError(f"Bilibili video lookup failed: {view.get('message')}")
    d = view["data"]
    pages = d.get("pages") or [{"cid": d["cid"], "duration": d.get("duration")}]
    pg = pages[min(max(page, 1), len(pages)) - 1]
    meta = {"title": d.get("title") + (f" (P{page})" if len(pages) > 1 else ""),
            "channel": (d.get("owner") or {}).get("name"), "duration": pg.get("duration") or d.get("duration"),
            "thumbnail": d.get("pic"), "bvid": d.get("bvid"), "page": page}
    return meta, pg


def bili_audio(s: requests.Session, bvid: str, cid: int) -> dict:
    """Audio-only DASH stream of a Bilibili video, lowest bitrate (plenty for speech)."""
    params = bili_sign(s, {"bvid": bvid, "cid": cid, "fnval": 16, "fnver": 0, "fourk": 0})
    pl = s.get("https://api.bilibili.com/x/player/wbi/playurl", params=params, timeout=15).json()
    audios = (((pl.get("data") or {}).get("dash") or {}).get("audio")) or []
    if not audios:
        raise ValueError(f"Bilibili returned no audio stream (code {pl.get('code')}: {pl.get('message')})")
    a = min(audios, key=lambda x: x.get("bandwidth") or 0)
    return {"url": a.get("baseUrl") or a.get("base_url"), "backup": a.get("backupUrl") or a.get("backup_url") or [],
            "id": a.get("id"), "kbps": round((a.get("bandwidth") or 0) / 1000),
            "headers": {"User-Agent": UA, "Referer": "https://www.bilibili.com/"}}


def bilibili_transcript(url: str, attempts: list) -> tuple[list[dict], dict] | None:
    s = bili_session()
    meta, pg = bili_video(s, url)
    try:
        params = bili_sign(s, {"bvid": meta["bvid"], "cid": pg["cid"]})
        pl = s.get("https://api.bilibili.com/x/player/wbi/v2", params=params, timeout=15).json()
        subs = ((pl.get("data") or {}).get("subtitle") or {}).get("subtitles") or []
        langs = [x.get("lan") for x in subs]
        fr = [x for x in subs if (x.get("lan") or "").lower().replace("ai-", "").startswith("fr")]
        if fr:
            x = sorted(fr, key=lambda x: (x.get("lan") or "").startswith("ai-"))[0]
            raw = s.get("https:" + x["subtitle_url"] if x["subtitle_url"].startswith("//") else x["subtitle_url"],
                        timeout=15).text
            lines = parse_subtitles(raw, "bili")
            if lines:
                return lines, dict(meta, lang=x.get("lan"), kind="ai" if x.get("lan", "").startswith("ai-") else "manual",
                                   method="bilibili-api")
        attempts.append({"method": "bilibili-api", "error": "no French subtitles", "available": langs,
                         "logged_in": bool(env("BILI_SESSDATA"))})
    except Exception as e:  # noqa: BLE001
        attempts.append({"method": "bilibili-api", "error": f"{type(e).__name__}: {str(e)[:300]}"})
    attempts.append({"meta": meta, "next": "no subtitle track: transcribe the audio (speech-to-text)"})
    return None


# ---------------------------------------------------------------------------
# 法语助手 (Eudic OpenAPI)
# ---------------------------------------------------------------------------

def eudic(method: str, path: str, **kw):
    tok = env("EUDIC_TOKEN")
    if not tok:
        raise ValueError("The 法语助手 token is not set on the helper (secret EUDIC_TOKEN).")
    auth = tok if tok.upper().startswith("NIS ") else f"NIS {tok}"
    r = requests.request(method, EUDIC_BASE + path, timeout=20,
                         headers={"Authorization": auth, "User-Agent": UA, "Content-Type": "application/json"}, **kw)
    if r.status_code == 401:
        raise ValueError("法语助手 rejected the token (401). Check EUDIC_TOKEN.")
    if r.status_code == 403:
        raise ValueError("法语助手 rate limit hit (403): max 30 calls/min, 500/day.")
    if r.status_code >= 400:
        raise ValueError(f"法语助手 error {r.status_code}: {r.text[:300]}")
    return r.json() if r.content else {"status": r.status_code}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(annotations=READ, structured_output=False)
def ping() -> dict:
    """Health check: helper version, which secrets are configured, and the helper's public IP."""
    try:
        ip = requests.get("https://api.ipify.org", timeout=5).text
    except requests.RequestException:
        ip = None
    return {"ok": True, "version": VERSION, "time": int(time.time()), "egress_ip": ip,
            "secrets": {"eudic": bool(env("EUDIC_TOKEN")),
                        "bilibili": bool(env("BILI_SESSDATA")),
                        "elevenlabs": bool(env("ELEVENLABS_API_KEY")),
                        "supadata": bool(env("SUPADATA_API_KEY")),
                        "groq": bool(env("GROQ_API_KEY"))},
            "host": "render" if os.environ.get("RENDER") else "other",
            "reflex_store": {_GitHubStore: "github", dict: "memory"}.get(type(_reflex_store()), "modal")}


@mcp.tool(annotations=READ, structured_output=False)
def get_transcript(url: str) -> dict:
    """Fetch the French subtitles of a YouTube or Bilibili video.

    Returns {ok, platform, id, url, title, channel, duration, lang, kind, method,
    lines: [{t: start seconds, d: duration seconds, text}], attempts}. When no French
    subtitles exist, ok is false and attempts says what was tried and what is available.
    """
    url = url.strip()
    attempts: list = []
    if "bilibili.com" in url or "b23.tv" in url or re.match(r"^(BV|av)\w+", url):
        if url.startswith(("BV", "av")):
            url = f"https://www.bilibili.com/video/{url}"
        got = bilibili_transcript(url, attempts)
        platform, vid = "bilibili", (re.search(r"(BV[0-9A-Za-z]{10})", url) or [None, None])[1]
    else:
        vid = youtube_id(url)
        if not vid:
            raise ValueError("Paste a YouTube or Bilibili video link.")
        platform = "youtube"
        meta = youtube_meta(vid)
        got = youtube_transcript(vid, attempts)
        if got:
            got = (got[0], dict(meta, **{k: v for k, v in got[1].items() if v}))
            got[1]["warning"] = caption_warning(got[1])
            got[1]["unusable"] = not str(got[1].get("lang") or "fr").startswith("fr")
        else:
            attempts.append({"meta": meta})
    if not got:
        meta = next((a["meta"] for a in attempts if isinstance(a, dict) and a.get("meta")), {})
        return {"ok": False, "platform": platform, "id": (meta.get("bvid") or vid), "url": url, **meta,
                "can_transcribe": platform == "bilibili" or bool(env("SUPADATA_API_KEY")), "attempts": attempts}
    lines, meta = got
    if platform == "bilibili":
        vid = meta.get("bvid") or vid
    return {"ok": True, "platform": platform, "id": vid, "url": url, **meta,
            "line_count": len(lines), "lines": lines, "attempts": attempts}


@mcp.tool(annotations=READ, structured_output=False)
def eudic_categories() -> dict:
    """List the 法语助手 生词本 (word-book) categories for French."""
    return eudic("GET", "/studylist/category", params={"language": "fr"})


@mcp.tool(annotations=READ, structured_output=False)
def eudic_words(category_id: str = "0", page: int = 1, page_size: int = 100) -> dict:
    """Words in one 法语助手 生词本 category (default 0 = 我的生词本), with their saved context lines."""
    return eudic("GET", "/studylist/words", params={"language": "fr", "category_id": category_id,
                                                     "page": page, "page_size": page_size})


@mcp.tool(annotations=READ, structured_output=False)
def eudic_word(word: str) -> dict:
    """Whether a word is in your 法语助手 生词本, and its saved details (404 → not there)."""
    try:
        return eudic("GET", "/studylist/word", params={"language": "fr", "word": word})
    except ValueError as e:
        if "404" in str(e):
            return {"found": False, "word": word}
        raise


@mcp.tool(annotations=READ, structured_output=False)
def eudic_recent(days: int = 30, page: int = 0, page_size: int = 100) -> dict:
    """Recently added 法语助手 entries (with 释义 and contexts) — also the ones you added while reading elsewhere."""
    return eudic("GET", "/studylist/vocab_entries", params={"language": "fr", "recent_days": days,
                                                            "page": page, "page_size": page_size})


@mcp.tool(annotations=READ, structured_output=False)
def eudic_mastered(page: int = 0, page_size: int = 100) -> dict:
    """Words you have marked as mastered in 法语助手."""
    return eudic("GET", "/studylist/mastered_words", params={"language": "fr", "page": page, "page_size": page_size})


@mcp.tool(annotations=READ, structured_output=False)
def eudic_notes(page: int = 0, page_size: int = 100) -> dict:
    """Your 法语助手 word notes."""
    return eudic("GET", "/studylist/notes", params={"language": "fr", "page": page, "page_size": page_size})


@mcp.tool(annotations=READ, structured_output=False)
def audio_probe(url: str) -> dict:
    """Check that the helper can download a video's audio track (YouTube or Bilibili): fetches the first 256 KB."""
    import yt_dlp
    try:
        if "bilibili.com" in url or "b23.tv" in url:
            s = bili_session()
            meta, pg = bili_video(s, url)
            a = bili_audio(s, meta["bvid"], pg["cid"])
            r = requests.get(a["url"], headers=dict(a["headers"], Range="bytes=0-262143"), timeout=20)
            return {"ok": r.status_code in (200, 206), "status": r.status_code, "fetched_bytes": len(r.content),
                    "kbps": a["kbps"], "duration": meta["duration"], "title": meta["title"]}
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "format": "bestaudio/best",
                               "extractor_args": {"youtube": {"player_client": ["android_vr", "default"]}}}) as ydl:
            info = ydl.extract_info(url.strip(), download=False)
        f = (info.get("requested_formats") or [info])[0]
        r = requests.get(f["url"], headers=dict(f.get("http_headers") or {}, Range="bytes=0-262143"), timeout=20)
        return {"ok": r.status_code in (200, 206), "status": r.status_code, "fetched_bytes": len(r.content),
                "format": f.get("format_id"), "ext": f.get("ext"), "abr": f.get("abr"),
                "size": f.get("filesize") or f.get("filesize_approx"), "duration": info.get("duration")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:400]}"}


def eleven(method: str, path: str, **kw) -> requests.Response:
    key = env("ELEVENLABS_API_KEY")
    if not key:
        raise ValueError("The ElevenLabs API key is not set on the helper (secret ELEVENLABS_API_KEY).")
    r = requests.request(method, "https://api.elevenlabs.io" + path, headers={"xi-api-key": key}, timeout=60, **kw)
    if r.status_code >= 400:
        raise ValueError(f"ElevenLabs error {r.status_code}: {r.text[:300]}")
    return r


@mcp.tool(annotations=READ, structured_output=False)
def elevenlabs_status() -> dict:
    """ElevenLabs plan and remaining credits, plus the voices on the account."""
    sub = eleven("GET", "/v1/user/subscription").json()
    voices = eleven("GET", "/v2/voices", params={"page_size": 100}).json().get("voices", [])
    return {"tier": sub.get("tier"), "characters_used": sub.get("character_count"),
            "characters_limit": sub.get("character_limit"), "resets_unix": sub.get("next_character_count_reset_unix"),
            "voices": [{"id": v.get("voice_id"), "name": v.get("name"), "category": v.get("category"),
                        "labels": v.get("labels")} for v in voices]}


@mcp.tool(annotations=READ, structured_output=False)
def tts_probe(text: str = "Quand bien même il pleuvrait, nous irions nous promener.",
              voice_id: str = "", fmt: str = "m4a") -> dict:
    """Speak a short French sentence with ElevenLabs. fmt "m4a" (AAC in MP4, storable in the app) or "mp3"; base64."""
    import tempfile
    voice_id = voice_id or env("ELEVENLABS_VOICE_ID") or "JBFqnCBsd6RMkjVDRZzb"
    r = eleven("POST", f"/v1/text-to-speech/{voice_id}", params={"output_format": "mp3_44100_64"},
               json={"text": text[:300], "model_id": "eleven_multilingual_v2", "language_code": "fr"})
    data, mime = r.content, "audio/mpeg"
    if fmt in ("m4a", "mp4v"):
        with tempfile.TemporaryDirectory() as tmp:
            open(f"{tmp}/in.mp3", "wb").write(data)
            to_storable(f"{tmp}/in.mp3", f"{tmp}/out.mp4", video=(fmt == "mp4v"))
            data, mime = open(f"{tmp}/out.mp4", "rb").read(), "video/mp4"
    return {"voice_id": voice_id, "mime": mime, "bytes": len(data), "audio_b64": base64.b64encode(data).decode()}


def _sentence_end_soon(words: list[dict], w: dict, ahead: int = 3) -> bool:
    """Does a sentence end within the next few words (counting w)? Then keep them on the current line."""
    try:
        i = words.index(w)
    except ValueError:
        return False
    nxt = [x for x in words[i:i + ahead + 3] if x.get("type") == "word"][:ahead]
    return any(re.search(r"[.?!…]$", x["text"]) for x in nxt)


def words_to_lines(words: list[dict], max_words: int = 14, max_gap: float = 0.8, keep_words: bool = False) -> list[dict]:
    """Group speech-to-text words into subtitle-like lines: break at sentence ends, long pauses or length."""
    lines, cur = [], []
    for w in words:
        if w.get("type") != "word":
            continue
        if cur and (w["start"] - cur[-1]["end"] > max_gap or len(cur) >= max_words + 4 or
                    (len(cur) >= max_words and not _sentence_end_soon(words, w))):
            lines.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.?!…]$", w["text"]):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    out = []
    for ws in lines:
        text = re.sub(r"\s+([,.?!…;:])", r"\1", " ".join(w["text"] for w in ws)).replace("' ", "'").replace("’ ", "’")
        ln = _line(ws[0]["start"], ws[-1]["end"], text)
        if ln:
            if keep_words:
                ln["w"] = [[w["text"], round(w["start"], 2), round(w["end"], 2)] for w in ws]
            out.append(ln)
    return out


def download(urls: list[str], headers: dict, dest: str, max_bytes: int | None = None) -> int:
    """Download the first URL that answers (Bilibili gives backups) to dest; returns the byte count."""
    errors = []
    for u in [x for x in urls if x]:
        try:
            h = dict(headers, Range=f"bytes=0-{max_bytes - 1}") if max_bytes else headers
            with requests.get(u, headers=h, stream=True, timeout=30) as r:
                if r.status_code not in (200, 206):
                    errors.append(f"HTTP {r.status_code}")
                    continue
                n = 0
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(256 * 1024):
                        fh.write(chunk)
                        n += len(chunk)
                return n
        except requests.RequestException as e:
            errors.append(type(e).__name__)
    raise ValueError(f"Audio download failed ({', '.join(errors) or 'no URL'})")


def ffmpeg(args: list[str]) -> None:
    import subprocess
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise ValueError(f"ffmpeg failed: {r.stderr.strip()[-400:]}")


def audio_seconds(path: str) -> float:
    import subprocess
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, timeout=60)
    return float(r.stdout.strip() or 0)


def to_storable(src: str, dest: str, video: bool = True, kbps: int = 64) -> None:
    """AAC in MP4. With video=True a 16x16 black 1 fps picture track is added, so the file is a real video
    (the artifact asset store accepts video/mp4 but refuses audio-only MP4)."""
    pic = ["-f", "lavfi", "-i", "color=c=black:s=16x16:r=1"] if video else []
    # -shortest overshoots with a 1 fps encoder, so cut the output at the audio's own duration.
    vcodec = ["-map", "1:a", "-map", "0:v", "-t", f"{audio_seconds(src):.3f}", "-c:v", "libx264",
              "-tune", "stillimage", "-pix_fmt", "yuv420p"] if video else []
    ffmpeg([*pic, "-i", src, *vcodec, "-ac", "1", "-c:a", "aac", "-b:a", f"{kbps}k",
            "-movflags", "+faststart", dest])


def cut_audio(src: str, seconds: float | None, dest: str) -> None:
    """Re-encode a local audio file to mono 48 kbps AAC, optionally keeping only the first `seconds`."""
    ffmpeg(["-i", src, "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "48k", *(["-t", str(seconds)] if seconds else []), dest])


def groq_stt(path: str) -> dict:
    """Groq Whisper: punctuated segments + word timings, reshaped like ElevenLabs output (words with type)."""
    with open(path, "rb") as fh:
        r = requests.post("https://api.groq.com/openai/v1/audio/transcriptions", timeout=600,
                          headers={"Authorization": f"Bearer {env('GROQ_API_KEY')}"},
                          data=[("model", "whisper-large-v3-turbo"), ("language", "fr"), ("response_format", "verbose_json"),
                                ("timestamp_granularities[]", "word"), ("timestamp_granularities[]", "segment")],
                          files={"file": ("audio.m4a", fh, "audio/mp4")})
    if r.status_code >= 400:
        raise ValueError(f"Groq speech-to-text error {r.status_code}: {r.content.decode('utf-8', 'replace')[:300]}")
    j = json.loads(r.content.decode("utf-8"))
    raw = [w for w in (j.get("words") or []) if str(w.get("word", "")).strip()]
    segs = j.get("segments") or []
    # Whisper's word list has no punctuation; borrow it from the segment text, word by word.
    out = []
    wi = 0
    for sg in segs:
        toks = str(sg.get("text", "")).split()
        for tok in toks:
            if wi >= len(raw):
                break
            w = raw[wi]
            if w["start"] > sg["end"] + 0.5:
                break
            out.append({"text": tok, "start": float(w["start"]), "end": float(w["end"]), "type": "word"})
            wi += 1
    for w in raw[wi:]:
        out.append({"text": str(w["word"]).strip(), "start": float(w["start"]), "end": float(w["end"]), "type": "word"})
    return {"words": out, "language_code": j.get("language") or "fr", "model": "groq-whisper-large-v3-turbo"}


def speech_to_text(path: str) -> dict:
    if env("GROQ_API_KEY"):
        try:
            return groq_stt(path)
        except ValueError:
            if not env("ELEVENLABS_API_KEY"):
                raise
    last = None
    for model in ("scribe_v2", "scribe_v1"):
        with open(path, "rb") as fh:
            r = requests.post("https://api.elevenlabs.io/v1/speech-to-text", timeout=600,
                              headers={"xi-api-key": env("ELEVENLABS_API_KEY")},
                              data={"model_id": model, "language_code": "fr", "timestamps_granularity": "word",
                                    "tag_audio_events": "false", "diarize": "false"},
                              files={"file": ("audio.m4a", fh, "audio/mp4")})
        if r.ok:
            return dict(r.json(), model=model)
        last = r
    raise ValueError(f"ElevenLabs speech-to-text error {last.status_code}: {last.text[:300]}")


@mcp.tool(annotations=READ, structured_output=False)
def stt_probe(url: str, seconds: int = 60) -> dict:
    """Transcribe the first `seconds` of a Bilibili video with ElevenLabs speech-to-text (tests quality and credit use)."""
    import tempfile
    s = bili_session()
    meta, pg = bili_video(s, url)
    a = bili_audio(s, meta["bvid"], pg["cid"])
    before = eleven("GET", "/v1/user/subscription").json().get("character_count")
    with tempfile.TemporaryDirectory() as tmp:
        src, dest = f"{tmp}/src.m4s", f"{tmp}/clip.m4a"
        t0 = time.time()
        download([a["url"], *a["backup"]], a["headers"], src)
        cut_audio(src, max(10, min(int(seconds), 600)), dest)
        t1 = time.time()
        res = speech_to_text(dest)
        t2 = time.time()
    after = eleven("GET", "/v1/user/subscription").json().get("character_count")
    lines = words_to_lines(res.get("words") or [])
    return {"title": meta["title"], "seconds": seconds, "model": res.get("model"),
            "language": res.get("language_code"), "cut_s": round(t1 - t0, 1), "stt_s": round(t2 - t1, 1),
            "credits_used": (after or 0) - (before or 0), "line_count": len(lines), "lines": lines}


# ---------------------------------------------------------------------------
# Jobs: work that can outlast one connector call (~130 s). The app starts a job, then polls job_status.
# State lives in a Modal Dict when running on Modal (any container can answer a poll), else in memory.
# ---------------------------------------------------------------------------

_JOBS: dict = {}


def _job_store():
    try:
        import modal
        if not modal.is_local():
            return modal.Dict.from_name("french-lab-jobs", create_if_missing=True)
    except ImportError:
        pass
    return _JOBS


def _job_set(jid: str, **fields) -> None:
    store = _job_store()
    cur = store.get(jid) or {}
    cur.update(fields, updated=time.time())
    store[jid] = cur


def _start_job(kind: str, fn, *args) -> dict:
    import threading
    import uuid
    jid = f"{kind}-{uuid.uuid4().hex[:12]}"
    _job_set(jid, kind=kind, state="running", step="starting", started=time.time())

    def run():
        try:
            result = fn(jid, *args)
            _job_set(jid, state="done", step="done", result=result)
        except Exception as e:  # noqa: BLE001 - the app shows the message
            _job_set(jid, state="error", step="failed", error=f"{type(e).__name__}: {str(e)[:500]}")

    threading.Thread(target=run, daemon=True).start()
    return {"job_id": jid, "state": "running"}


def bili_audio_file(url: str, dest: str) -> dict:
    """Download a Bilibili video's audio track to dest; returns the video's meta."""
    s = bili_session()
    meta, pg = bili_video(s, url)
    a = bili_audio(s, meta["bvid"], pg["cid"])
    download([a["url"], *a["backup"]], a["headers"], dest)
    return meta


def _transcribe(jid: str, url: str) -> dict:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _job_set(jid, step="downloading audio")
        meta = bili_audio_file(url, f"{tmp}/src.m4s")
        _job_set(jid, step="preparing audio")
        cut_audio(f"{tmp}/src.m4s", None, f"{tmp}/a.m4a")
        _job_set(jid, step="transcribing")
        res = speech_to_text(f"{tmp}/a.m4a")
    lines = words_to_lines(res.get("words") or [])
    return {"ok": True, "platform": "bilibili", "id": meta["bvid"], "url": url, **meta,
            "lang": res.get("language_code"), "kind": "speech-to-text", "method": res.get("model"),
            "line_count": len(lines), "lines": lines}


def _transcribe_youtube(jid: str, url: str) -> dict:
    vid = youtube_id(url)
    _job_set(jid, step="transcribing the audio (Supadata AI)")
    j = _supadata_get({"url": f"https://www.youtube.com/watch?v={vid}", "lang": "fr", "mode": "generate",
                       "chunkSize": 110}, 540)
    lines, lang = _supadata_lines(j)
    lines = resegment(lines)
    if not lines:
        raise ValueError("The AI transcript came back empty.")
    meta = youtube_meta(vid)
    return {"ok": True, "platform": "youtube", "id": vid, "url": url, **meta, "lang": lang or "fr",
            "kind": "ai-transcript", "method": "supadata-generate", "line_count": len(lines), "lines": lines}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True), structured_output=False)
def start_transcription(url: str) -> dict:
    """Start an AI transcription of a video's audio. Bilibili: ElevenLabs. YouTube: Supadata AI
    (about 2 Supadata credits per minute). Poll job_status."""
    url = url.strip()
    if "bilibili.com" in url or "b23.tv" in url or url.startswith(("BV", "av")):
        return _start_job("stt", _transcribe, url)
    if not youtube_id(url):
        raise ValueError("Paste a YouTube or Bilibili video link.")
    return _start_job("stt", _transcribe_youtube, url)


@mcp.tool(annotations=READ, structured_output=False)
def job_status(job_id: str) -> dict:
    """State of a job: {state: running|done|error, step, result?, error?}."""
    j = _job_store().get(job_id)
    if not j:
        return {"job_id": job_id, "state": "unknown"}
    return dict(j, job_id=job_id)


# ---------------------------------------------------------------------------
# Audio for dictée: real clips (Bilibili) and ElevenLabs speech. Both come back as small video-wrapped
# MP4s (base64) because the app's file store refuses audio-only files.
# ---------------------------------------------------------------------------

@mcp.tool(annotations=READ, structured_output=False)
def audio_clips(url: str, segments: list[dict], pad: float = 0.25) -> dict:
    """Cut clips of a Bilibili video's own audio. segments: [{t: start s, d: duration s}] (max 40, 30 s each)."""
    import tempfile
    segs = segments[:40]
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        bili_audio_file(url.strip(), f"{tmp}/src.m4s")
        for i, sg in enumerate(segs):
            t0 = max(0.0, float(sg["t"]) - pad)
            d = min(30.0, float(sg["d"]) + 2 * pad)
            ffmpeg(["-ss", f"{t0:.2f}", "-t", f"{d:.2f}", "-i", f"{tmp}/src.m4s", "-vn", "-ac", "1",
                    "-c:a", "aac", "-b:a", "48k", f"{tmp}/c{i}.m4a"])
            to_storable(f"{tmp}/c{i}.m4a", f"{tmp}/c{i}.mp4", video=True, kbps=48)
            data = open(f"{tmp}/c{i}.mp4", "rb").read()
            out.append({"t": sg["t"], "d": sg["d"], "bytes": len(data), "b64": base64.b64encode(data).decode()})
    return {"mime": "video/mp4", "clips": out}


@mcp.tool(annotations=READ, structured_output=False)
def speak(text: str, voice_id: str = "", slow: bool = False) -> dict:
    """Read French text aloud with ElevenLabs (uses character credits). Returns a video-wrapped MP4, base64."""
    import tempfile
    voice_id = voice_id or env("ELEVENLABS_VOICE_ID") or "JBFqnCBsd6RMkjVDRZzb"
    body = {"text": text[:600], "model_id": "eleven_multilingual_v2", "language_code": "fr"}
    if slow:
        body["voice_settings"] = {"speed": 0.8, "stability": 0.6, "similarity_boost": 0.75}
    r = eleven("POST", f"/v1/text-to-speech/{voice_id}", params={"output_format": "mp3_44100_64"}, json=body)
    with tempfile.TemporaryDirectory() as tmp:
        open(f"{tmp}/in.mp3", "wb").write(r.content)
        to_storable(f"{tmp}/in.mp3", f"{tmp}/out.mp4", video=True)
        data = open(f"{tmp}/out.mp4", "rb").read()
    return {"voice_id": voice_id, "mime": "video/mp4", "chars": len(body["text"]), "bytes": len(data),
            "b64": base64.b64encode(data).decode()}


_PIPER: dict = {}


@mcp.tool(annotations=READ, structured_output=False)
def read_aloud(text: str) -> dict:
    """Read French text aloud with Piper, a free open-source voice running on the helper (no credits).
    Returns a video-wrapped MP4, base64, for the app's file store."""
    import tempfile
    import wave
    text = str(text or "").strip()[:600]
    if not text:
        raise ValueError("Nothing to read.")
    if "voice" not in _PIPER:
        from piper import PiperVoice
        _PIPER["voice"] = PiperVoice.load(os.environ.get("PIPER_VOICE", "/voices/fr_FR-siwis-medium.onnx"))
    voice = _PIPER["voice"]
    with tempfile.TemporaryDirectory() as tmp:
        with wave.open(f"{tmp}/in.wav", "wb") as wf:
            (voice.synthesize_wav if hasattr(voice, "synthesize_wav") else voice.synthesize)(text, wf)
        to_storable(f"{tmp}/in.wav", f"{tmp}/out.mp4", video=True, kbps=48)
        data = open(f"{tmp}/out.mp4", "rb").read()
    return {"voice": "piper fr_FR-siwis", "mime": "video/mp4", "bytes": len(data), "b64": base64.b64encode(data).decode()}


# ---------------------------------------------------------------------------
# 法语助手 sync. Limits: 30 calls/min, 500/day → paced inside a job.
# ---------------------------------------------------------------------------

class _Pacer:
    def __init__(self, per_min: int = 26):
        self.gap = 60.0 / per_min
        self.last = 0.0

    def call(self, *a, **kw):
        wait = self.last + self.gap - time.time()
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        return eudic(*a, **kw)


def _eudic_sync(jid: str, items: list[dict], category: str) -> dict:
    p = _Pacer()
    cats = p.call("GET", "/studylist/category", params={"language": "fr"}).get("data") or []
    cat = next((c for c in cats if c.get("name") == category), None)
    if not cat:
        p.call("POST", "/studylist/category", json={"language": "fr", "name": category})
        cats = p.call("GET", "/studylist/category", params={"language": "fr"}).get("data") or []
        cat = next((c for c in cats if c.get("name") == category), None)
    if not cat:
        raise ValueError(f"Could not create the 生词本 « {category} »")
    cid = cat["id"]
    done = []
    for n, it in enumerate(items[:60]):
        _job_set(jid, step=f"adding {n + 1}/{len(items)}: {it['word']}")
        body = {"language": "fr", "word": it["word"], "star": int(it.get("star") or 3), "category_ids": [cid]}
        if it.get("context"):
            body["context_line"] = it["context"][:500]
        try:
            p.call("POST", "/studylist/word", json=body)
            row = {"word": it["word"], "ok": True}
            if it.get("note"):
                try:
                    old = p.call("GET", "/studylist/note", params={"language": "fr", "word": it["word"]})
                    old_note = (old.get("data") or old).get("note", "") if isinstance(old, dict) else ""
                    old_note = re.sub(r"<!--meta.*?-->", "", old_note or "", flags=re.S).strip()
                except ValueError:
                    old_note = ""
                if it["note"] not in (old_note or ""):
                    note = (old_note + "\n\n" if old_note else "") + it["note"]
                    p.call("POST", "/studylist/note", json={"language": "fr", "word": it["word"], "note": note[:2000]})
                    row["note"] = True
        except ValueError as e:
            row = {"word": it["word"], "ok": False, "error": str(e)[:200]}
        done.append(row)
    # Read back 法语助手's own 释义 for what was added (one paged call).
    _job_set(jid, step="reading back 释义")
    exp = {}
    try:
        words = p.call("GET", "/studylist/words",
                       params={"language": "fr", "category_id": cid, "page": 1, "page_size": 100}).get("data") or []
        exp = {w["word"]: w.get("exp") for w in words}
    except ValueError:
        pass
    for row in done:
        row["exp"] = exp.get(row["word"])
    return {"category": category, "category_id": cid, "items": done}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True), structured_output=False)
def eudic_sync(items: list[dict], category: str = "French Lab") -> dict:
    """Add expressions to a 法语助手 生词本 (created if missing), each with its context line and an optional note.
    items: [{word, context?, star? 1-5, note?}] (max 60). Runs as a job (rate limits) — poll job_status."""
    clean = [i for i in items if isinstance(i, dict) and str(i.get("word", "")).strip()]
    if not clean:
        raise ValueError("Nothing to sync.")
    return _start_job("eudic", _eudic_sync, clean, category.strip() or "French Lab")



# ---------------------------------------------------------------------------
# Podcasts: search (Apple Podcasts directory), episode lists (RSS), and a job that downloads an episode,
# stores it as ≤30-minute speech-quality parts (video-wrapped MP4, ~0.3 MB/min) and transcribes it with
# word timings. The app then fetches each part (podcast_part) into its own file store.
# ---------------------------------------------------------------------------

DATA_DIR = os.environ.get("FL_DATA_DIR", "/tmp/fl-data")
PART_SECONDS = 600  # 10-minute parts (~3 MB): larger uploads from a phone to the artifact store fail
NS = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd", "podcast": "https://podcastindex.org/namespace/1.0",
      "content": "http://purl.org/rss/1.0/modules/content/"}


def _volume():
    try:
        import modal
        if not modal.is_local():
            return modal.Volume.from_name("french-lab-data", create_if_missing=True)
    except ImportError:
        pass
    return None


def _duration(v) -> int:
    v = str(v or "").strip()
    if not v:
        return 0
    if ":" in v:
        try:
            return int(_ts(v))
        except ValueError:
            return 0
    try:
        return int(float(v))
    except ValueError:
        return 0


@mcp.tool(annotations=READ, structured_output=False)
def podcast_search(term: str, limit: int = 12) -> dict:
    """Search podcasts by name or topic (Apple Podcasts directory, French store)."""
    r = requests.get("https://itunes.apple.com/search", timeout=20, headers={"User-Agent": UA},
                     params={"term": term, "media": "podcast", "entity": "podcast", "limit": max(1, min(limit, 25)),
                             "country": "FR"})
    out = []
    for x in json.loads(r.content.decode("utf-8")).get("results", []):
        if x.get("feedUrl"):
            out.append({"id": x.get("collectionId"), "name": x.get("collectionName"), "author": x.get("artistName"),
                        "feed": x.get("feedUrl"), "episodes": x.get("trackCount"), "genre": x.get("primaryGenreName"),
                        "latest": (x.get("releaseDate") or "")[:10]})
    return {"results": out}


def _resolve_feed(url: str) -> tuple[str, str | None]:
    """Feed URL for an RSS link or an Apple Podcasts link (+ the Apple episode id when the link has ?i=)."""
    m = re.search(r"podcasts\.apple\.com/.*?/id(\d+)", url)
    if m:
        ep = (re.search(r"[?&]i=(\d+)", url) or [None, None])[1]
        j = json.loads(requests.get("https://itunes.apple.com/lookup", params={"id": m.group(1)}, timeout=20).content)
        feed = (j.get("results") or [{}])[0].get("feedUrl")
        if not feed:
            raise ValueError("Apple Podcasts doesn't list a public feed for this show.")
        return feed, ep
    return url, None


def _parse_feed(xml: bytes, limit: int) -> dict:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    ch = root.find("channel")
    if ch is None:
        raise ValueError("That link isn't a podcast feed.")
    pod = {"title": (ch.findtext("title") or "").strip(), "author": (ch.findtext("itunes:author", namespaces=NS) or "").strip(),
           "link": (ch.findtext("link") or "").strip(), "language": (ch.findtext("language") or "").strip()}
    eps = []
    for it in ch.findall("item"):
        enc = it.find("enclosure")
        if enc is None or not enc.get("url"):
            continue
        tr = None
        for t in it.findall("podcast:transcript", NS):
            typ = (t.get("type") or "").lower()
            if any(k in typ for k in ("vtt", "subrip", "srt", "json")):
                tr = {"url": t.get("url"), "type": typ}
                break
        desc = it.findtext("itunes:summary", namespaces=NS) or it.findtext("description") or ""
        eps.append({"guid": (it.findtext("guid") or enc.get("url")).strip(), "title": (it.findtext("title") or "").strip(),
                    "date": (it.findtext("pubDate") or "").strip(), "duration": _duration(it.findtext("itunes:duration", namespaces=NS)),
                    "audio": enc.get("url"), "bytes": int(enc.get("length") or 0) if (enc.get("length") or "").isdigit() else 0,
                    "link": (it.findtext("link") or "").strip(), "desc": _clean(desc)[:400], "transcript": tr})
        if len(eps) >= limit:
            break
    return {"podcast": pod, "episodes": eps}


@mcp.tool(annotations=READ, structured_output=False)
def podcast_episodes(url: str, limit: int = 40) -> dict:
    """Episodes of a podcast, from its RSS feed URL or an Apple Podcasts link (newest first)."""
    feed, apple_ep = _resolve_feed(url.strip())
    r = requests.get(feed, timeout=30, headers={"User-Agent": UA})
    if r.status_code >= 400:
        raise ValueError(f"The feed answered HTTP {r.status_code}.")
    out = _parse_feed(r.content, max(1, min(limit, 200)))
    out["feed"] = feed
    if apple_ep:
        try:
            j = json.loads(requests.get("https://itunes.apple.com/lookup", timeout=20,
                                        params={"id": re.search(r"/id(\d+)", url).group(1), "entity": "podcastEpisode",
                                                "limit": 200}).content)
            hit = next((x for x in j.get("results", []) if str(x.get("trackId")) == apple_ep), None)
            if hit:
                out["focus"] = {"title": hit.get("trackName"), "audio": hit.get("episodeUrl")}
        except Exception:  # noqa: BLE001 - the plain list still works
            pass
    return out


def _parse_transcript(url: str, typ: str) -> list[dict]:
    raw = requests.get(url, timeout=30, headers={"User-Agent": UA}).content.decode("utf-8", "replace")
    if "json" in typ:
        j = json.loads(raw)
        segs = j.get("segments") or []
        return [ln for sg in segs if (ln := _line(float(sg.get("startTime", 0)), float(sg.get("endTime", 0)), sg.get("body", "")))]
    return parse_subtitles(raw, "vtt" if "vtt" in typ else "srt")


def _podcast(jid: str, audio_url: str, transcript_url: str, transcript_type: str) -> dict:
    import shutil
    base = os.path.join(DATA_DIR, jid)
    os.makedirs(base, exist_ok=True)
    # tidy up parts older than two days
    try:
        for d in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, d)
            if d != jid and os.path.isdir(path) and time.time() - os.path.getmtime(path) > 2 * 86400:
                shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass
    src = os.path.join(base, "src")
    _job_set(jid, step="downloading the episode")
    n = download([audio_url], {"User-Agent": UA}, src)
    if n < 10_000:
        raise ValueError("The episode audio couldn't be downloaded.")
    dur = audio_seconds(src)
    if not dur:
        raise ValueError("That file doesn't look like audio.")
    parts = []
    k = 0
    while k * PART_SECONDS < dur - 1:
        start = k * PART_SECONDS
        length = min(PART_SECONDS, dur - start)
        _job_set(jid, step=f"compressing part {k + 1} of {int((dur - 1) // PART_SECONDS) + 1}")
        out = os.path.join(base, f"part{k}.mp4")
        ffmpeg(["-f", "lavfi", "-i", "color=c=black:s=16x16:r=1", "-ss", f"{start:.2f}", "-t", f"{length:.2f}", "-i", src,
                "-map", "1:a", "-map", "0:v", "-t", f"{length:.2f}", "-c:v", "libx264", "-tune", "stillimage",
                "-pix_fmt", "yuv420p", "-ac", "1", "-c:a", "aac", "-b:a", "40k", "-movflags", "+faststart", out])
        parts.append({"n": k, "start": round(start, 2), "dur": round(length, 2), "bytes": os.path.getsize(out)})
        k += 1
    if transcript_url:
        _job_set(jid, step="reading the publisher's transcript")
        lines = _parse_transcript(transcript_url, transcript_type)
        source = "publisher transcript"
    else:
        _job_set(jid, step="transcribing (" + ("Groq Whisper" if env("GROQ_API_KEY") else "ElevenLabs") + ")")
        stt = os.path.join(base, "stt.m4a")
        ffmpeg(["-i", src, "-vn", "-ac", "1", "-c:a", "aac", "-b:a", "32k", stt])
        res = speech_to_text(stt)
        lines = words_to_lines(res.get("words") or [], keep_words=True)
        source = f"{res.get('model')} speech-to-text"
    os.remove(src)
    vol = _volume()
    if vol:
        vol.commit()
    return {"duration": round(dur, 2), "parts": parts, "lines": lines, "line_count": len(lines), "source": source}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=True), structured_output=False)
def start_podcast(audio_url: str, transcript_url: str = "", transcript_type: str = "") -> dict:
    """Download a podcast episode, store it as speech-quality parts and transcribe it (word timings) — a job.
    Poll job_status; then fetch each part with podcast_part. Uses ElevenLabs speech-to-text unless a
    publisher transcript (VTT/SRT/JSON) is given."""
    return _start_job("pod", _podcast, audio_url.strip(), transcript_url.strip(), transcript_type.strip())


@mcp.tool(annotations=READ, structured_output=False)
def podcast_part(job_id: str, n: int) -> dict:
    """One stored part of a podcast job (video-wrapped MP4, base64), for the app's file store."""
    path = os.path.join(DATA_DIR, job_id, f"part{int(n)}.mp4")
    if not os.path.exists(path):
        vol = _volume()
        if vol:
            vol.reload()
    if not os.path.exists(path):
        raise ValueError("That part is no longer on the helper — start the episode again.")
    data = open(path, "rb").read()
    return {"n": int(n), "mime": "video/mp4", "bytes": len(data), "b64": base64.b64encode(data).decode()}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, openWorldHint=False), structured_output=False)
def podcast_cleanup(job_id: str) -> dict:
    """Delete a podcast job's stored parts from the helper (after the app has saved them)."""
    import shutil
    if not re.fullmatch(r"pod-[0-9a-f]{12}", job_id):
        raise ValueError("Not a podcast job id.")
    shutil.rmtree(os.path.join(DATA_DIR, job_id), ignore_errors=True)
    vol = _volume()
    if vol:
        vol.commit()
    return {"ok": True}


@mcp.tool(annotations=READ, structured_output=False)
def payload_probe(kb: int = 512) -> dict:
    """Return `kb` kilobytes of random base64 data with its SHA-256 (tests how much a result can carry)."""
    kb = max(1, min(int(kb), 24 * 1024))
    raw = os.urandom(kb * 1024)
    return {"kb": kb, "sha256": hashlib.sha256(raw).hexdigest(), "data": base64.b64encode(raw).decode()}


# ---------------------------------------------------------------------------
# Réflexes: whatever the user says in a Claude chat reaches the Réflexes app through this inbox
# ---------------------------------------------------------------------------
_REFLEX: dict = {}
REFLEX_WRITE = ToolAnnotations(readOnlyHint=False, openWorldHint=False)
GRADES = ("again", "hard", "good", "easy")


class _GitHubStore:
    """A small dict kept as JSON files in a private GitHub repo (branch "data"), because free hosts have no lasting
    disk and the inbox must survive restarts. "in:*" keys share reflexes/inbox.json; any other key gets its own file.
    One instance is the only writer, so memory is the truth and every change is written straight through."""

    API = "https://api.github.com"
    BRANCH = "data"

    def __init__(self, repo: str, token: str):
        self.repo, self.files, self.sha, self.branch_ok = repo, {}, {}, False
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                               "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "french-lab-helper"})
        self.lock = threading.Lock()

    @staticmethod
    def _group(key) -> str:
        return "inbox" if str(key).startswith("in:") else re.sub(r"[^\w-]", "_", str(key))

    def _url(self, g: str) -> str:
        return f"{self.API}/repos/{self.repo}/contents/reflexes/{g}.json"

    def _load(self, g: str) -> dict:
        if g not in self.files:
            r = self.s.get(self._url(g), params={"ref": self.BRANCH}, timeout=20)
            if r.status_code == 404:
                self.files[g], self.sha[g] = {}, None
            else:
                r.raise_for_status()
                j = r.json()
                raw = base64.b64decode(j["content"]) if j.get("content") else self.s.get(
                    self._url(g), params={"ref": self.BRANCH}, headers={"Accept": "application/vnd.github.raw+json"},
                    timeout=30).content  # over 1 MB the JSON reply carries no content
                self.files[g], self.sha[g] = json.loads(raw or b"{}"), j["sha"]
        return self.files[g]

    def _ensure_branch(self) -> None:
        if self.branch_ok:
            return
        ref = f"{self.API}/repos/{self.repo}/git/ref/heads/{self.BRANCH}"
        if self.s.get(ref, timeout=20).status_code == 404:
            repo = self.s.get(f"{self.API}/repos/{self.repo}", timeout=20)
            repo.raise_for_status()
            head = self.s.get(f"{self.API}/repos/{self.repo}/git/ref/heads/{repo.json()['default_branch']}", timeout=20)
            head.raise_for_status()
            r = self.s.post(f"{self.API}/repos/{self.repo}/git/refs", timeout=20,
                            json={"ref": f"refs/heads/{self.BRANCH}", "sha": head.json()["object"]["sha"]})
            if r.status_code != 422:  # 422: it was created in the meantime
                r.raise_for_status()
        self.branch_ok = True

    def _save(self, g: str) -> None:
        self._ensure_branch()
        body = {"message": f"reflexes {g}", "branch": self.BRANCH,
                "content": base64.b64encode(json.dumps(self.files[g], ensure_ascii=False).encode()).decode()}
        for attempt in range(2):
            if self.sha.get(g):
                body["sha"] = self.sha[g]
            r = self.s.put(self._url(g), json=body, timeout=30)
            if r.status_code in (409, 422) and attempt == 0:  # stale sha (file changed elsewhere): take the new one
                cur = self.s.get(self._url(g), params={"ref": self.BRANCH}, timeout=20)
                self.sha[g] = cur.json().get("sha") if cur.ok else None
                continue
            r.raise_for_status()
            self.sha[g] = r.json()["content"]["sha"]
            return

    def get(self, key, default=None):
        with self.lock:
            return self._load(self._group(key)).get(key, default)

    def __setitem__(self, key, value) -> None:
        self.update({key: value})

    def update(self, items: dict) -> None:
        with self.lock:
            groups = set()
            for k, v in items.items():
                g = self._group(k)
                self._load(g)[k] = v
                groups.add(g)
            for g in groups:
                self._save(g)

    def pop(self, key, default=None):
        with self.lock:
            g = self._group(key)
            d = self._load(g)
            if key not in d:
                return default
            v = d.pop(key)
            self._save(g)
            return v

    def pop_many(self, keys: list) -> int:
        with self.lock:
            groups, n = set(), 0
            for k in keys:
                g = self._group(k)
                if self._load(g).pop(k, None) is not None:
                    groups.add(g)
                    n += 1
            for g in groups:
                self._save(g)
            return n

    def items(self):
        with self.lock:
            return list(self._load("inbox").items())


_GH_STORE: list = []


def _reflex_store():
    if env("GH_STORE_TOKEN") and env("GH_STORE_REPO"):
        if not _GH_STORE:
            _GH_STORE.append(_GitHubStore(env("GH_STORE_REPO"), env("GH_STORE_TOKEN")))
        return _GH_STORE[0]
    try:
        import modal
        if not modal.is_local():
            return modal.Dict.from_name("french-lab-reflexes", create_if_missing=True)
    except ImportError:
        pass
    return _REFLEX


def _inbox_put(*entries: dict) -> list[str]:
    """Add entries to the inbox in one write; returns their ids."""
    batch = {}
    for entry in entries:
        eid = f"{int(time.time() * 1000):x}{os.urandom(2).hex()}"
        batch[f"in:{eid}"] = {**entry, "id": eid, "at": int(time.time() * 1000)}
    if batch:
        _reflex_store().update(batch)
    return [v["id"] for v in batch.values()]


def _clip(v, n: int) -> str:
    return str(v or "").strip()[:n]


@mcp.tool(annotations=REFLEX_WRITE, structured_output=False)
def reflexes_save(said: str, french: str = "", note: str = "", expressions: list[str] | None = None) -> dict:
    """Save something from this conversation to the user's Réflexes notebook — their phone app that files everyday
    spoken French on their knowledge map and drills it until it comes out on reflex. Call it, without asking, whenever
    the user shares French they heard, asks how to say something, tries out a sentence, or asks you to keep/note
    something. said = what the user said, word for word; french = the natural spoken French you gave them;
    note = one short English line (meaning, correction, or when to use it); expressions = 1-3 short expressions worth
    drilling. The app organises it the next time it is opened."""
    if not _clip(said, 2000):
        raise ValueError("Nothing to save.")
    [eid] = _inbox_put({"kind": "save", "said": _clip(said, 2000), "french": _clip(french, 1000), "note": _clip(note, 600),
                        "expressions": [_clip(x, 120) for x in (expressions or []) if _clip(x, 120)][:5]})
    return {"saved": True, "id": eid}


@mcp.tool(annotations=READ, structured_output=False)
def reflexes_due(limit: int = 8) -> dict:
    """Expressions from the user's Réflexes notebook to quiz them on in this chat, due ones first, then new ones.
    Each: id, t (the French), en, zh, sit (situation), cue_en / cue_zh (a moment where they'd say it), say (example),
    prompt (a line someone might say to them) and replies (natural answers using it). Quiz one at a time: give the cue
    (in English unless they prefer Chinese), let them answer out loud or by dictation, then give the French and a short
    comment. Report each outcome with reflexes_result."""
    snap = _reflex_store().get("snapshot") or {}
    items = list(snap.get("items") or [])
    t = time.time() * 1000
    due = sorted([i for i in items if i.get("due") and i["due"] <= t], key=lambda i: i["due"])
    fresh = [i for i in items if not i.get("due")]
    pick = (due + fresh)[: max(1, min(int(limit), 20))]
    return {"updated": snap.get("at"), "total": len(items), "due": len(due), "items": pick,
            "note": None if items else "The notebook is empty or the app hasn't been opened since this was set up."}


@mcp.tool(annotations=REFLEX_WRITE, structured_output=False)
def reflexes_result(results: list[dict]) -> dict:
    """Record how the user did on expressions quizzed in this chat: [{id, grade: again|hard|good|easy, said: what they
    answered}]. again = blank or wrong, hard = got there slowly, good = right after a moment, easy = instantly.
    The app updates its review schedule from these."""
    ids = _inbox_put(*[{"kind": "result", "expr": _clip(r["id"], 40), "grade": r["grade"], "said": _clip(r.get("said"), 300)}
                       for r in (results or [])[:50] if r.get("id") and r.get("grade") in GRADES])
    return {"recorded": len(ids)}


@mcp.tool(annotations=READ, structured_output=False)
def reflexes_inbox() -> dict:
    """(For the Réflexes app.) Everything saved from chats since the app last collected it."""
    st = _reflex_store()
    out = [v for k, v in st.items() if str(k).startswith("in:")]
    out.sort(key=lambda e: e.get("at", 0))
    return {"entries": out[:300]}


@mcp.tool(annotations=REFLEX_WRITE, structured_output=False)
def reflexes_ack(ids: list[str]) -> dict:
    """(For the Réflexes app.) Remove inbox entries it has collected."""
    st = _reflex_store()
    keys = [f"in:{i}" for i in (ids or [])[:500]]
    if hasattr(st, "pop_many"):
        return {"removed": st.pop_many(keys)}
    return {"removed": sum(st.pop(k, None) is not None for k in keys)}


@mcp.tool(annotations=REFLEX_WRITE, structured_output=False)
def reflexes_sync(items: list[dict]) -> dict:
    """(For the Réflexes app.) Store the notebook's expression list so chats can quiz from it."""
    keep = ("id", "t", "en", "zh", "sit", "cue_en", "cue_zh", "say", "prompt", "replies", "status", "due")
    clean = [{k: i.get(k) for k in keep if i.get(k) not in (None, "", [])} for i in (items or [])[:2000] if i.get("id") and i.get("t")]
    _reflex_store()["snapshot"] = {"at": int(time.time() * 1000), "items": clean}
    return {"stored": len(clean)}


@mcp.custom_route("/", methods=["GET"])
async def home(_request):
    from starlette.responses import PlainTextResponse
    return PlainTextResponse("French Lab helper is running.")


def _keep_awake() -> None:
    """A free Render service sleeps after 15 minutes without inbound traffic and takes about a minute to wake —
    longer than claude.ai waits for a connector. A request to our own public address every 10 minutes counts as
    traffic (one always-on service fits in Render's 750 free hours a month). KEEP_AWAKE=0 turns it off."""
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url or os.environ.get("KEEP_AWAKE") == "0":
        return

    def loop():
        while True:
            time.sleep(600)
            try:
                requests.get(url.rstrip("/") + "/", timeout=30)
            except requests.RequestException:
                pass

    threading.Thread(target=loop, daemon=True).start()


app = mcp.streamable_http_app()
_keep_awake()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
