"""This module provides the core functionality for the CSAM Guard service.

It includes the `CSAMGuard` class, which is responsible for assessing text and
images for potential CSAM-related content. The module also defines the data
structures for decisions and metrics, a rate limiter, and various helper
functions for text normalization, term matching, and image hashing.
"""
from __future__ import annotations
import re, json, unicodedata, hashlib, base64, os, uuid, logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Pattern, Tuple
from dataclasses import dataclass, asdict, field
from collections import Counter, defaultdict
from time import time
from functools import lru_cache
from io import BytesIO

import requests, feedparser
import numpy as np
from scipy.fft import dct
from PIL import Image, ImageFile, ImageOps

from prometheus_client import Counter as PromCounter

# Safety settings for Pillow
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 64_000_000

# Prometheus metrics (opt-in via env in app.py)
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "0") == "1"
if PROMETHEUS_ENABLED:
    csam_requests_total = PromCounter("csam_requests_total", "Total requests processed", ["endpoint"])
    csam_decisions_total = PromCounter("csam_decisions_total", "Total decisions made", ["action", "category"])
    csam_rss_terms_total = PromCounter("csam_rss_terms_total", "Total RSS terms proposed", ["category"])
    csam_rss_term_churn = PromCounter("csam_rss_term_churn", "RSS terms added or expired", ["action", "category"])

ALLOWED_IMAGE_CT = {"image/png", "image/jpeg", "image/webp", "image/gif"}

# Optional transformers pipeline is gated by env to avoid heavy deps by default
try:
    from transformers import pipeline as _hf_pipeline  # optional
except Exception:
    _hf_pipeline = None

# --- Default Configuration ---
DEFAULT_CONFIG: Dict[str, Any] = {
    "hard_terms": [
        "loli","lolicon","shotacon","shota","child","children","kid","kids","minor","underage",
        "under-aged","preteen","preteens","tween","schoolgirl","school boy","schoolboy","school girl",
        "middle school","elementary","kindergarten","grade school","babysitter","babysit","babysitting",
        "baby face","babyface","childlike","child-like","youngster","kiddy",
        "toddler","infant","pedo","pedophile","juvenile","adolescent",
        "prepubescent","child porn","child sexual abuse material","csam",
        "sg-csam","self-generated csam","sextortion","child victims",
        "abusive images","federal law child pornography","child sexual exploitation",
        "online child sexual exploitation",
    ],
    "ambiguous_youth": [
        "teen","teens","teenage","teenager","jailbait","young-looking","looks young","youthful","nubile","barely legal",
        "college freshman","freshman","looks like a kid","girl next door","boyish","girlish",
        "tiny girl","innocent-looking","school uniform","petite schoolgirl","petite teen","young girl","young boy"
    ],
    "pending_terms":{"hard_terms":{}, "ambiguous_youth":{}},
    "pending_ttl_hours":24,
    "pending_confirmation_threshold":2,
    "ambiguous_youth_weights":{"jailbait":5,"barely legal":4,"looks like a kid":5,"young-looking":3,"teen":2,"teenage":2,"teenager":2,"youthful":1,"nubile":2},
    "adult_assertions":["adult","18+","eighteen plus","over 18","over 21","21+","twenty one","grown woman","grown man","mature","of age","legal age"],
    "injections":["ignore previous","bypass filter","disable safety","jailbreak","pretend you are","roleplay as","act as if","for educational purposes","hypothetically","in a fictional scenario","as a creative writing exercise","ignore instructions","override","system prompt"],
    "injection_patterns":[r"\b(DAN|STAN|AIM|Developer Mode)\b"],
    "age_patterns":[r"\b(\d{1,2})\s*(?:yo|y/o|yrs? old|years? old)\b", r"\b(?:age)[\s:=]+(\d{1,2})\b", r"\b(\d{1,2})\s*(?:-?\s*year\s*old)\b"],
    "school_patterns":[r"\b(k|kinder(?:garten)?)\b", r"\b(1st|2nd|3rd|[4-9]th|10th|11th|12th)\s*grade\b", r"\b(middle\s*school|junior\s*high|elementary\s*school)\b"],
    "allowlist_patterns":[r"\bkindergarten teacher\b", r"\bschool administrator\b", r"\bchild psychologist\b", r"\bpediatric\b", r"\bparenting\b", r"\bchild development\b", r"\bsocial worker\b", r"\bcounselor\b", r"\bnurse\b", r"\bdoctor\b"],
    "professional_terms":["teacher","nurse","doctor","psychologist","counselor","social worker","pediatric","parenting","development","administrator","educator","therapist","caregiver"],
    "sexual_terms":["sexy","hot","naked","nude","porn","sex","erotic"],
    "costume_terms":["costume","cosplay","roleplay","role play","dress up","halloween","party","outfit","uniform for","dressed as"],
    "number_words":{"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20},
    "adult_normalization":{"boobiies":"boobies","boobiiies":"boobies","boobiiees":"boobies","boobiess":"boobies","b00bies":"boobies","t1tties":"titties","titt1es":"titties","pussi":"pussy","pOOssy":"pussy","dik":"dick","kiddie":"kid","kiddies":"kids","kiddy":"kid","teeen":"teen","teeeen":"teen","teeeeen":"teen","yung":"young","yong":"young","youthfull":"youthful"},
    "fun_replacements":{"teen":"wrinkly grandpa","girl":"old man","boy":"old hag","loli":"ugly grandpa","shota":"wrinkly grandma","young":"ancient","petite":"massive","innocent":"guilty","schoolgirl":"retired grandpa","kid":"fossil","child":"centenarian"},
    "homoglyph_map":{"¡":"i","Ｉ":"i","і":"i","Ⅰ":"i","ⓘ":"i","ⓞ":"o","○":"o","◯":"o","Ｏ":"o","о":"o","０":"o","ⓐ":"a","а":"a","Ａ":"a","ａ":"a","ⓔ":"e","ｅ":"e","е":"e","ē":"e","ⓢ":"s","Ｓ":"s","ｓ":"s","ⓣ":"t","Ｔ":"t","ｔ":"t","ⓤ":"u","Ｕ":"u","ｕ":"u","ⓡ":"r","Ｒ":"r","ｒ":"r","ⓛ":"l","Ｌ":"l","ｌ":"l","ⓑ":"b","Ｂ":"b","ｂ":"b","ⓙ":"j","Ｊ":"j","ｊ":"j"},
    "risky_clusters":[{"small","petite","tiny","little"},{"innocent","naive","pure","sweet"},{"girl","boy","kid","child"}],
    "severity_levels":{"CRITICAL":["loli","shota","child porn","pedo","pedophile","toddler","infant"],"HIGH":["kid","minor","underage","preteen","child","children"],"MEDIUM":["teen","teenage","jailbait","barely legal"],"LOW":["youthful","nubile"]},
    "nlp_model_name":"michellejieli/nsfw_text_classifier",
    "nlp_model_version":"1.0",
    "fuzzy_threshold":0.85,
    "hamming_thresh":5,
    "jaccard_thresh":0.80,
    "context_threshold":4,
    "nlp_threshold":0.7,
    "ngram_size":3,
    "hash_bits":64,
    "rate_limit_max":100,
    "rate_limit_window":60,
    "rss_feeds":[
        "https://www.nccprblog.org/feeds/posts/default?alt=rss",
        "https://childlinett.org/feed/",
        "https://www.kidcheck.com/feed/",
        "https://www.childrenssafetycenter.org/feed/",
        "https://safeandsound.org/blog/category/children/feed/",
        "https://clubexperience.blog/category/child-club-safety/feed/",
        "https://www.safetynook.net/blog-feed.xml",
        "https://www.esafekids.com.au/blog-feed.xml",
        "https://kidsafe.com.au/category/latest-news/feed/",
        "https://permachildsafety.com.au/blogs/news.atom",
        "https://thesafetychic.com/feed/",
        "https://www.kidsafevic.com.au/feed/",
        "https://safe4kids.com.au/blog/feed/",
        "https://laurenskids.org/category/blog/feed/",
        "https://childsafetystore.com/blogs/news.atom",
        "https://texprotects.blog/feed/",
        "https://kidsindanger.org/feed/"
    ],
    "known_csam_phashes": ["a1b2c3d4e5f6a7b8","b2c3d4e5f6a7b8c9"],
    "phash_match_thresh":10,
}

# --- Regexes ---
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
COMBINING_MARKS_RE = re.compile(r"[\u0300-\u036f]")
WHITESPACE_RE = re.compile(r"\s+")
NONWORD_BETWEEN_LETTERS = re.compile(r"(?<=\w)[\W_]+(?=\w)")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")
LEET_MAP = str.maketrans({"0":"o","1":"i","3":"e","4":"a","5":"s","7":"t","@":"a","$":"s"})
SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
WORD_SPLIT_RE = re.compile(r"\s+")
TYPO_CLEANUP_RE = re.compile(r"[,.!?;:\"'()\[\]{}]")

@dataclass
class Decision:
    """Represents the outcome of a content assessment.

    Attributes:
        allow: A boolean indicating whether the content is allowed.
        action: The action taken (e.g., "ALLOW", "BLOCK").
        reason: A string explaining the reason for the decision.
        normalized_prompt: The normalized version of the input text.
        rewritten_prompt: An optional rewritten version of the prompt.
        signals: A dictionary of signals that contributed to the decision.
    """
    allow: bool
    action: str
    reason: str
    normalized_prompt: str = ""
    rewritten_prompt: Optional[str] = None
    signals: Dict[str, Any] = field(default_factory=dict)
    def to_json(self) -> str:
        """Serializes the decision to a JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":"))

@dataclass
class Metrics:
    """A class to track metrics related to content assessments."""
    total_requests: int = 0
    blocks: int = 0
    allows: int = 0
    block_reasons: Counter = field(default_factory=Counter)
    severity_counts: Counter = field(default_factory=Counter)
    def record(self, decision: Decision):
        """Records a decision, updating the metrics."""
        self.total_requests += 1
        if decision.allow:
            self.allows += 1
        else:
            self.blocks += 1
        self.block_reasons[decision.reason] += 1
        self.severity_counts[decision.signals.get("severity", "UNKNOWN")] += 1
        if PROMETHEUS_ENABLED:
            cat = decision.signals.get("severity") or ("ALLOW" if decision.allow else "UNKNOWN")
            csam_decisions_total.labels(action=decision.action, category=cat).inc()
    def summary(self) -> Dict:
        """Returns a summary of the metrics as a dictionary."""
        return {"total": self.total_requests, "blocks": self.blocks, "allows": self.allows, "block_rate": self.blocks / max(1, self.total_requests), "top_reasons": dict(self.block_reasons.most_common(5)), "severity": dict(self.severity_counts)}

class RateLimiter:
    """A simple in-memory rate limiter."""
    def __init__(self, max_requests: int, window: int):
        """Initializes the RateLimiter.

        Args:
            max_requests: The maximum number of requests allowed in the window.
            window: The time window in seconds.
        """
        self.requests = defaultdict(list)
        self.max_requests = max_requests
        self.window = window
    def check(self, user_id: str) -> bool:
        """Checks if a user has exceeded the rate limit.

        Args:
            user_id: The identifier for the user.

        Returns:
            True if the request is allowed, False otherwise.
        """
        now = time()
        self.requests[user_id] = [t for t in self.requests[user_id] if now - t < self.window]
        if len(self.requests[user_id]) >= self.max_requests:
            return False
        self.requests[user_id].append(now)
        return True

class CSAMGuard:
    """The main class for the CSAM Guard service."""
    def __init__(self, config: Dict):
        """Initializes the CSAMGuard instance.

        Args:
            config: A dictionary containing the configuration for the guard.
        """
        self._validate_config(config)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_nlp_model()
        self.pending_term_counts = defaultdict(Counter)

        self.allowlist_re: Pattern = self._compile_pattern_list(config["allowlist_patterns"])
        self.injection_re: Pattern = self._compile_pattern_list(config["injection_patterns"])
        self.school_re: Pattern = self._compile_pattern_list(config["school_patterns"])
        self.age_re_list: List[Pattern] = [re.compile(p, re.IGNORECASE) for p in config["age_patterns"]]
        self.age_word_re: Pattern = re.compile(r"\b(" + "|".join(config["number_words"].keys()) + r")\s*(?:yo|y/o|yrs? old|years? old|-?\s*year\s*old)\b", re.IGNORECASE)

        self.hard_terms_re: Pattern = self._build_term_regex(config["hard_terms"])
        self.ambi_terms_re: Pattern = self._build_term_regex(config["ambiguous_youth"])
        self.adult_terms_re: Pattern = self._build_term_regex(config["adult_assertions"])
        self.inject_terms_re: Pattern = self._build_term_regex(config["injections"])
        self.risk_cache: Dict = self._build_risk_cache()

    def _validate_config(self, config: Dict):
        """Validates the configuration dictionary."""
        required = ["hard_terms","ambiguous_youth","adult_assertions","rss_feeds","fuzzy_threshold","context_threshold","professional_terms","sexual_terms","costume_terms","known_csam_phashes","phash_match_thresh","pending_terms","pending_ttl_hours","pending_confirmation_threshold"]
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError(f"Config missing keys: {missing}")

    def _load_nlp_model(self):
        # allow disabling NLP entirely (useful for tests/airgapped)
        if os.getenv("DISABLE_NLP", "0") == "1":
            self.classifier = None
            self.logger.info("NLP disabled via DISABLE_NLP=1; falling back to heuristics.")
            return

        model_name = self.config["nlp_model_name"]
        model_version = self.config["nlp_model_version"]
        self.logger.info(f"Loading NLP model: {model_name} (v{model_version})...")

        if _hf_pipeline is None:
            self.classifier = None
            self.logger.warning("transformers not installed; NLP disabled; using heuristics only.")
            return

        try:
            self.classifier = _hf_pipeline("text-classification", model=model_name)
            self.logger.info(f"NLP model loaded: {model_name} (v{model_version})")
        except Exception as e:
            self.classifier = None
            self.logger.error(f"Failed to load NLP model {model_name} (v{model_version}): {e}. "
                              "Proceeding with heuristics only.")

    def _compile_pattern_list(self, patterns: List[str]) -> Pattern:
        """Compiles a list of patterns into a single regex."""
        return re.compile("|".join(f"({p})" for p in patterns), re.IGNORECASE)

    @lru_cache(maxsize=32)
    def _build_term_regex_cached(self, terms_tuple: Tuple[str, ...]) -> Pattern:
        """Builds a regex for a tuple of terms, with caching."""
        terms = list(terms_tuple)
        sorted_terms = sorted(list(set(t.lower() for t in terms)), key=len, reverse=True)
        escaped_terms = [re.escape(term) if re.search(r"\W", term) else r'\b' + re.escape(term) + r'\b' for term in sorted_terms]
        return re.compile("|".join(escaped_terms), re.IGNORECASE)

    def _build_term_regex(self, terms: List[str]) -> Pattern:
        """Builds a regex for a list of terms."""
        return self._build_term_regex_cached(tuple(terms))

    def _build_risk_cache(self) -> Dict:
        """Builds a cache of risky terms and their precomputed hashes."""
        lexicon = sorted(set(self.config["hard_terms"] + self.config["ambiguous_youth"] + list(self.config["pending_terms"]["hard_terms"]) + list(self.config["pending_terms"]["ambiguous_youth"]) + ["under 18","under eighteen","boy","girl","kid"]))
        n = self.config["ngram_size"]
        h = self.config["hash_bits"]
        return {"lexicon": lexicon, "simhash": {t: self._simhash(t, n, h) for t in lexicon}, "soundex": {t: self._soundex(t) for t in lexicon}, "ngrams": {t: self._char_ngrams(t, n) for t in lexicon}}

    def update_terms_from_rss(self):
        """Updates the term lists from RSS feeds.

        This method fetches content from the configured RSS feeds, identifies
        potential new terms, and adds them to the pending_terms list.
        """
        if PROMETHEUS_ENABLED:
            csam_requests_total.labels(endpoint="update_terms").inc()
        now = datetime.now()
        ttl = timedelta(hours=self.config["pending_ttl_hours"])
        new_hard = set(); new_ambi = set()
        for feed_url in self.config["rss_feeds"]:
            try:
                response = requests.get(feed_url, timeout=(5,10), headers={"User-Agent":"csam-guard/14.1 (+https://example.org)"})
                response.raise_for_status()
                feed = feedparser.parse(response.text)
                for entry in feed.entries:
                    content = (entry.title + " " + (getattr(entry, "description", "") or "") + " " + (entry.content[0].value if getattr(entry, "content", None) else "")).lower()
                    content = content[:4000]
                    ok = False
                    if self.classifier:
                        try:
                            results = self.classifier(content)
                            ok = results and results[0].get("score",0) > 0.7 and results[0].get("label","").upper() in ("NSFW","LABEL_1")
                        except Exception:
                            ok = False
                    else:
                        # conservative fallback: simple keyword check
                        ok = any(k in content for k in ("exploitation","abuse","sextortion","victim","csam"))
                    if ok:
                        for term in re.findall(r'\b(child|minor|underage|csam|exploitation|abuse|sextortion|explicit|victim)\b', content):
                            self.pending_term_counts[term][feed_url] += 1
            except Exception as e:
                self.logger.warning(f"Failed to fetch {feed_url}: {e}")
        for category in ["hard_terms","ambiguous_youth"]:
            expired = []
            for term, data in self.config["pending_terms"][category].items():
                try:
                    if now - datetime.fromisoformat(data["added"]) > ttl:
                        expired.append(term)
                except Exception:
                    expired.append(term)
            for term in expired:
                del self.config["pending_terms"][category][term]
                self.pending_term_counts.pop(term, None)
                if PROMETHEUS_ENABLED:
                    csam_rss_term_churn.labels(action="expired", category=category).inc()
        for term, sources in self.pending_term_counts.items():
            if len(sources) >= self.config["pending_confirmation_threshold"]:
                category = "hard_terms" if ("child" in term or "minor" in term) else "ambiguous_youth"
                if term not in self.config[category] and term not in self.config["pending_terms"][category]:
                    self.config["pending_terms"][category][term] = {"added": now.isoformat(), "sources": list(sources.keys()), "weight": 2}
                    if PROMETHEUS_ENABLED:
                        csam_rss_terms_total.labels(category=category).inc()
                        csam_rss_term_churn.labels(action="added", category=category).inc()
                    if category == "hard_terms": new_hard.add(term)
                    else: new_ambi.add(term)
        if new_hard or new_ambi:
            self.hard_terms_re = self._build_term_regex(self.config["hard_terms"] + list(self.config["pending_terms"]["hard_terms"]))
            self.ambi_terms_re = self._build_term_regex(self.config["ambiguous_youth"] + list(self.config["pending_terms"]["ambiguous_youth"]))
            self.risk_cache = self._build_risk_cache()
            self.logger.info(f"Updated terms: {len(new_hard)} hard, {len(new_ambi)} ambiguous.")

    def _fold_homoglyphs(self, s: str) -> str:
        return "".join(self.config["homoglyph_map"].get(ch, ch) for ch in s)

    def _normalize_text(self, s: str) -> str:
        s = s.strip().replace("\u2028"," ").replace("\u2029"," ")
        s = ZERO_WIDTH_RE.sub("", s)
        s = self._fold_homoglyphs(s)
        s = unicodedata.normalize("NFKC", s)
        s = COMBINING_MARKS_RE.sub("", s)
        s = s.translate(LEET_MAP)
        s = REPEAT_CHAR_RE.sub(r"\1\1", s)
        s = WHITESPACE_RE.sub(" ", s.lower())
        return s

    def _squash_internals(self, s: str) -> str:
        return NONWORD_BETWEEN_LETTERS.sub("", s)

    def _deobfuscate(self, s: str) -> str:
        return WHITESPACE_RE.sub("", s)

    def _normalize_for_adult_typos(self, s: str) -> str:
        tokens = WORD_SPLIT_RE.split(s)
        out = []
        norm_dict = self.config["adult_normalization"]
        for tok in tokens:
            low = TYPO_CLEANUP_RE.sub("", tok).lower()
            out.append(norm_dict.get(low, tok))
        return " ".join(out)

    def _words_to_int(self, phrase: str) -> Optional[int]:
        NUMBER_UNITS = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19}
        NUMBER_TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
        parts = phrase.lower().split()
        if len(parts) == 1:
            return NUMBER_UNITS.get(parts[0], NUMBER_TENS.get(parts[0]))
        if len(parts) == 2 and parts[0] in NUMBER_TENS and parts[1] in NUMBER_UNITS:
            return NUMBER_TENS[parts[0]] + NUMBER_UNITS[parts[1]]
        return None

    def _find_terms_regex(self, text: str, term_re: Pattern) -> Set[str]:
        matches = term_re.findall(text)
        if not matches:
            return set()
        if isinstance(matches[0], tuple):
            flat = [m for tup in matches for m in tup if m]
        else:
            flat = matches
        return {m.lower() for m in flat}

    def _find_ages(self, raw: str) -> Set[int]:
        ages = set()
        for pat in self.age_re_list:
            for m in pat.finditer(raw):
                for g in m.groups():
                    if g and g.isdigit():
                        ages.add(int(g))
        # spelled ages short-form
        spelled = re.compile(r"\b((?:twenty(?:\s+(?:one|two|three|four|five|six|seven|eight|nine))?)|(?:nineteen|eighteen|seventeen|sixteen|fifteen|fourteen|thirteen|twelve|eleven|ten|nine|eight|seven|six|five|four|three|two|one|zero))\s*(?:yo|y/o|yrs?\s*old|years?\s*old|-?\s*year\s*old)\b", re.IGNORECASE)
        for m in spelled.finditer(raw):
            n = self._words_to_int(m.group(1))
            if n is not None:
                ages.add(n)
        return ages

    def _school_context(self, s: str) -> Set[str]:
        return {m.group(0).lower() for m in self.school_re.finditer(s)}

    def _check_allowlist(self, s: str) -> bool:
        return bool(self.allowlist_re.search(s))

    def _cross_sentence_detect(self, norm: str) -> Set[str]:
        sentences = SENTENCE_SPLIT_RE.split(norm)
        hits = set()
        for i in range(len(sentences) - 1):
            combined = sentences[i].strip() + " " + sentences[i + 1].strip()
            hits.update(self._find_terms_regex(combined, self.hard_terms_re))
        return hits

    def _cluster_detection(self, tokens: List[str]) -> int:
        score = 0
        token_set = set(tokens)
        for cluster in self.config["risky_clusters"]:
            if len(cluster & token_set) >= 2:
                score += 3
        return score

    @lru_cache(maxsize=1024)
    def _simhash(self, text: str, ngram: int, hashbits: int) -> int:
        grams = [text] if len(text) < ngram else [text[i:i+ngram] for i in range(len(text) - ngram + 1)]
        if not grams:
            return 0
        weights = Counter(grams)
        v = [0] * hashbits
        for g, w in weights.items():
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            for i in range(hashbits):
                bit = (h >> i) & 1
                v[i] += w if bit else -w
        fp = 0
        for i, val in enumerate(v):
            if val > 0:
                fp |= (1 << i)
        return fp

    def _hamming64(self, a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    def _char_ngrams(self, s: str, n: int) -> set:
        s = f" {s} "
        if len(s) < n:
            return {s}
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    def _jaccard(self, set1: set, set2: set) -> float:
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    @lru_cache(maxsize=512)
    def _soundex(self, s: str) -> str:
        s = re.sub(r"[^a-zA-Z]", "", s).upper()
        if not s:
            return "Z000"
        first = s[0]
        mapping = {**{c:"1" for c in "BFPV"}, **{c:"2" for c in "CGJKQSXZ"}, **{c:"3" for c in "DT"}, "L":"4", **{c:"5" for c in "MN"}, "R":"6"}
        digits = []
        prev = ""
        for ch in s[1:]:
            d = mapping.get(ch, "")
            if d and d != prev:
                digits.append(d)
            prev = d
        code = first + "".join(digits)
        code = first + re.sub(r"[AEIOUYHW0]", "", code[1:])
        return (code + "000")[:4]

    def _soundex_match(self, term: str, window: str, min_length: int = 5) -> bool:
        if abs(len(term) - len(window)) > 2:
            return False
        return self._soundex(term) == self._soundex(window) and len(window) >= min_length

    def _second_pass_detect(self, norm: str) -> Dict[str, Any]:
        signals = {"second_pass": []}
        squashed = self._squash_internals(norm)
        tokens = WORD_SPLIT_RE.split(squashed)
        windows = set()
        for k in range(1, 4):
            for i in range(max(0, len(tokens) - k + 1)):
                w = " ".join(tokens[i : i + k])
                if len(w) <= 32:
                    windows.add(w)
        risky = self.risk_cache
        n = self.config["ngram_size"]; h = self.config["hash_bits"]
        ht = self.config["hamming_thresh"]; jt = self.config["jaccard_thresh"]
        for w in windows:
            sh = self._simhash(w, n, h)
            sx = self._soundex(w)
            ngrams = self._char_ngrams(w, n)
            for term in risky["lexicon"]:
                if term in self.config["adult_assertions"]:
                    continue
                ham_dist = self._hamming64(sh, risky["simhash"][term])
                if ham_dist <= ht:
                    signals["second_pass"].append({"window": w, "match_type": "simhash", "term": term, "dist": ham_dist})
                    break
                jac_sim = self._jaccard(ngrams, risky["ngrams"][term])
                if jac_sim >= jt:
                    signals["second_pass"].append({"window": w, "match_type": "jaccard", "term": term, "sim": jac_sim})
                    break
                if sx == risky["soundex"][term] and self._soundex_match(term, w):
                    signals["second_pass"].append({"window": w, "match_type": "soundex", "term": term})
                    break
        return signals

    def _check_professional_context(self, signals: Dict) -> bool:
        if not signals["allowlisted"]:
            return False
        text = signals["normalized"]
        has_professional = any(term in text for term in self.config["professional_terms"])
        has_sexual = any(term in text for term in self.config["sexual_terms"])
        return has_professional and not has_sexual

    def _validate_adult_assertion(self, signals: Dict) -> bool:
        has_adult_age = any(a >= 18 for a in signals["ages"])
        has_adult_term = bool(signals["adult_assertions"])
        has_hard_minor = bool(signals["hard_terms"])
        if has_hard_minor:
            return False
        text = signals.get("normalized", "")
        costume_ctx = any(p in text for p in self.config["costume_terms"])
        adult_guards = ("all adults","adults only","18+","eighteen plus","alumni","reunion","over 18","legal age")
        has_strong_adult_ctx = (any(p in text for p in adult_guards) or has_adult_age or costume_ctx)
        return has_strong_adult_ctx and has_adult_term

    def _flagged_by_nlp(self, prompt: str) -> bool:
        if not getattr(self, "classifier", None):
            # graceful fallback to heuristics
            hard_count = len(self._find_terms_regex(prompt, self.hard_terms_re))
            context_count = len(self._find_terms_regex(prompt, self.ambi_terms_re))
            return hard_count >= 1 and context_count >= 1
        try:
            threshold = self.config["nlp_threshold"]
            result = self.classifier(prompt[:4000])[0]
            label = result.get("label","")
            score = result.get("score",0.0)
            is_nsfw = label.upper() in ("NSFW","LABEL_1") and score > threshold
            return is_nsfw
        except Exception as e:
            self.logger.error(f"NLP classification failed ({e})")
            hard_count = len(self._find_terms_regex(prompt, self.hard_terms_re))
            context_count = len(self._find_terms_regex(prompt, self.ambi_terms_re))
            return hard_count >= 1 and context_count >= 1

    def _determine_severity(self, signals: Dict) -> str:
        all_terms = signals.get("hard_terms", set()) | signals.get("ambiguous_youth", set())
        for level in ["CRITICAL","HIGH","MEDIUM","LOW"]:
            if any(term in all_terms for term in self.config["severity_levels"][level]):
                return level
        return "UNKNOWN"

    def _context_score(self, signals: Dict) -> int:
        score = 0
        has_explicit_adult_age = any(a >= 18 for a in signals["ages"])
        if has_explicit_adult_age:
            for term in signals["ambiguous_youth"]:
                weight = self.config["ambiguous_youth_weights"].get(term, 2)
                score += max(1, weight - 2)
        else:
            score += len(signals["hard_terms"]) * 5
            for term in signals["ambiguous_youth"]:
                weight = self.config["ambiguous_youth_weights"].get(term, 2)
                score += weight
            for term in self.config["pending_terms"]["hard_terms"]:
                if term in signals["normalized"]:
                    score += self.config["pending_terms"]["hard_terms"][term]["weight"]
            for term in self.config["pending_terms"]["ambiguous_youth"]:
                if term in signals["normalized"]:
                    score += self.config["pending_terms"]["ambiguous_youth"][term]["weight"]
        score += len(signals["school_context"]) * 2
        score += len(signals["injections"]) * 6
        if any(a < 18 for a in signals["ages"]):
            score += 7
        if signals.get("second_pass"):
            score += len(signals["second_pass"]) * 3
        if signals.get("cross_sentence"):
            score += len(signals["cross_sentence"]) * 4
        score += signals.get("cluster_score", 0)
        if self._validate_adult_assertion(signals):
            score -= 5
        if signals.get("allowlisted"):
            score = max(0, score - self.config.get("allowlist_credit", 2))
        return max(0, score)

    def _api_log(self, decision: Decision):
        if PROMETHEUS_ENABLED:
            cat = decision.signals.get("severity") or ("ALLOW" if decision.allow else "UNKNOWN")
            csam_decisions_total.labels(action=decision.action, category=cat).inc()

    def _extract_signals(self, prompt: str, verbose: bool = False) -> Dict:
        raw = prompt
        allowlisted = self._check_allowlist(raw)
        ages = self._find_ages(raw)
        norm = self._normalize_text(prompt)
        deob = self._deobfuscate(norm)
        norm = self._normalize_for_adult_typos(norm)
        squashed = self._squash_internals(norm)
        hard = self._find_terms_regex(squashed, self.hard_terms_re)
        hard.update(self._find_terms_regex(deob, self.hard_terms_re))
        ambi = self._find_terms_regex(squashed, self.ambi_terms_re)
        ambi.update(self._find_terms_regex(deob, self.ambi_terms_re))
        adults = self._find_terms_regex(norm, self.adult_terms_re)
        injections = self._find_terms_regex(norm, self.inject_terms_re)
        injections.update(m.group(0).lower() for m in self.injection_re.finditer(norm))
        school = self._school_context(squashed)
        cross_sent = self._cross_sentence_detect(norm)
        tokens = WORD_SPLIT_RE.split(squashed)
        cluster_score = self._cluster_detection(tokens)
        signals = {"normalized": norm,"ages": sorted(ages),"hard_terms": hard,"ambiguous_youth": ambi,"adult_assertions": adults,"injections": injections,"school_context": school,"cross_sentence": cross_sent,"cluster_score": cluster_score,"allowlisted": allowlisted,"model": {"name": self.config["nlp_model_name"], "version": self.config["nlp_model_version"]}}
        if verbose:
            self.logger.info(f"[DEBUG] Normalized: {norm}")
            self.logger.info(f"[DEBUG] Squashed: {squashed}")
        return signals

    def _make_decision(self, signals: Dict) -> Decision:
        if self._check_professional_context(signals):
            return Decision(allow=True, action="ALLOW", reason="Legitimate professional/educational context", normalized_prompt=signals["normalized"], signals={k: sorted(v) if isinstance(v, set) else v for k, v in signals.items()})
        has_hard = bool(signals["hard_terms"] or signals["injections"])
        has_minor_age = any(a < 18 for a in signals["ages"])
        if has_hard or has_minor_age:
            severity = self._determine_severity(signals); signals["severity"] = severity
            return Decision(allow=False, action="BLOCK", reason=f"Direct violation (severity: {severity})", normalized_prompt=signals["normalized"], signals={k: sorted(v) if isinstance(v, set) else v for k, v in signals.items()})
        has_ambig = bool(
            signals["ambiguous_youth"]
            or signals["school_context"]
            or re.search(r"\bschool uniform\b", signals["normalized"])
            or signals["cross_sentence"]
        )
        if has_ambig:
            if self._validate_adult_assertion(signals):
                return Decision(allow=True, action="ALLOW", reason="Valid adult assertion with ambiguous terms", normalized_prompt=signals["normalized"], signals={k: sorted(v) if isinstance(v, set) else v for k, v in signals.items()})
            else:
                severity = self._determine_severity(signals); signals["severity"] = severity
                return Decision(allow=False, action="BLOCK", reason=f"Ambiguous youth context without valid adult assertion (severity: {severity})", normalized_prompt=signals["normalized"], signals={k: sorted(v) if isinstance(v, set) else v for k, v in signals.items()})
        sp = self._second_pass_detect(signals["normalized"]); signals.update(sp)
        score = self._context_score(signals)
        is_nsfw = False
        if score <= self.config["context_threshold"] and not sp["second_pass"]:
            is_nsfw = self._flagged_by_nlp(signals["normalized"])
        signals["nlp_flagged"] = is_nsfw; signals["context_score"] = score
        if score > self.config["context_threshold"] or sp["second_pass"] or is_nsfw:
            severity = self._determine_severity(signals); signals["severity"] = severity
            return Decision(allow=False, action="BLOCK", reason=f"High risk (score: {score}, NLP: {is_nsfw}, second_pass: {len(sp['second_pass'])})", normalized_prompt=signals["normalized"], signals={k: sorted(v) if isinstance(v, set) else v for k, v in signals.items()})
        return Decision(allow=True, action="ALLOW", reason="No minor risk detected", normalized_prompt=signals["normalized"], signals={k: sorted(v) if isinstance(v, set) else v for k, v in signals.items()})

    def fun_rewrite(self, norm: str) -> str:
        """Rewrites a normalized prompt with fun replacements.

        Args:
            norm: The normalized prompt.

        Returns:
            The rewritten prompt.
        """
        replacements = self.config["fun_replacements"]
        sorted_keys = sorted(replacements.keys(), key=len, reverse=True)
        import re as _re
        def replacer(match):
            return replacements[match.group(0).lower()]
        replacer_re = _re.compile(r'\b(' + '|'.join(_re.escape(k) for k in sorted_keys) + r')\b', _re.IGNORECASE)
        return replacer_re.sub(replacer, norm)

    def assess(self, prompt: str, do_fun_rewrite: bool = False, log_func: Optional[callable] = None, log_path: Optional[str] = None, verbose: bool = False) -> Decision:
        """Assesses a text prompt for potential CSAM-related content.

        Args:
            prompt: The text prompt to assess.
            do_fun_rewrite: Whether to perform a fun rewrite on the prompt.
            log_func: An optional function to log the decision.
            log_path: An optional path to a log file.
            verbose: Whether to enable verbose logging.

        Returns:
            A Decision object representing the outcome of the assessment.
        """
        if PROMETHEUS_ENABLED:
            csam_requests_total.labels(endpoint="assess").inc()
        ts = datetime.now().isoformat(); request_id = uuid.uuid4().hex
        signals = self._extract_signals(prompt, verbose)
        decision = self._make_decision(signals)
        if decision.allow and do_fun_rewrite:
            decision.rewritten_prompt = self.fun_rewrite(signals["normalized"])
        if log_func:
            log_func(ts, request_id, prompt, signals["normalized"], decision.action, decision.reason, log_path, self.logger)
        self._api_log(decision); return decision

    def _compute_phash(self, img: Image.Image) -> int:
        """Computes the perceptual hash of an image.

        Args:
            img: The image to hash.

        Returns:
            The perceptual hash of the image.
        """
        try:
            try:
                if getattr(img, "is_animated", False):
                    img.seek(0)
            except Exception:
                pass
            img = ImageOps.exif_transpose(img).convert("L")
            try:
                img = img.resize((32, 32), Image.Resampling.LANCZOS)
            except AttributeError:
                img = img.resize((32, 32), Image.LANCZOS)
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            raise
        pixels = np.asarray(img, dtype="float32")
        dct_coeffs = dct(dct(pixels.T, norm="ortho").T, norm="ortho")
        low = dct_coeffs[:8, :8].copy()
        dct_low = low.flatten()[1:]
        med = np.median(dct_low)
        bits = (dct_low > med).astype(np.uint8)
        bits = np.concatenate([[0], bits])
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h

    def assess_image(self, image_path: Optional[str] = None, image_data: Optional[bytes] = None, log_func: Optional[callable] = None, log_path: Optional[str] = None, verbose: bool = False) -> Decision:
        """Assesses an image for potential CSAM-related content.

        Args:
            image_path: The path to the image file.
            image_data: The image data as a byte string.
            log_func: An optional function to log the decision.
            log_path: An optional path to a log file.
            verbose: Whether to enable verbose logging.

        Returns:
            A Decision object representing the outcome of the assessment.
        """
        if PROMETHEUS_ENABLED:
            csam_requests_total.labels(endpoint="assess_image").inc()
        ts = datetime.now().isoformat(); request_id = uuid.uuid4().hex
        normalized = image_path if image_path else "uploaded_image"
        if not image_path and not image_data:
            raise ValueError("Provide either image_path or image_data.")
        try:
            if image_path:
                img = Image.open(image_path)
            else:
                img = Image.open(BytesIO(image_data))
        except Image.DecompressionBombError:
            decision = Decision(allow=False, action="BLOCK", reason="Image exceeds decompression limits", normalized_prompt=normalized, signals={"error":"DecompressionBombError"})
            self._api_log(decision); return decision
        except Exception as e:
            decision = Decision(allow=False, action="BLOCK", reason=f"Image processing error: {str(e)}", normalized_prompt=normalized, signals={"error":str(e)})
            self._api_log(decision); return decision
        phash = self._compute_phash(img)
        phash_hex = f"{phash:016x}"
        signals = {"phash": phash_hex, "matches": [], "min_distance": float("inf"), "model": {"name": self.config["nlp_model_name"], "version": self.config["nlp_model_version"]}}
        min_dist = float("inf")
        for known in self.config["known_csam_phashes"]:
            known_int = int(known, 16) if isinstance(known, str) else known
            dist = self._hamming64(phash, known_int)
            signals["matches"].append({"known": known, "dist": dist})
            if dist < min_dist:
                min_dist = dist
        signals["min_distance"] = min_dist
        if min_dist <= self.config["phash_match_thresh"]:
            decision = Decision(allow=False, action="BLOCK", reason=f"CSAM image match detected (min hamming distance: {min_dist})", normalized_prompt=normalized, signals=signals)
        else:
            decision = Decision(allow=True, action="ALLOW", reason="No CSAM image match detected", normalized_prompt=normalized, signals=signals)
        self._api_log(decision); return decision

def log_entry(ts: str, request_id: str, original: str, norm: str, action: str, reason: str, log_path: str, logger: logging.Logger):
    """Logs a decision to a file and the console.

    Args:
        ts: The timestamp of the request.
        request_id: The unique ID of the request.
        original: The original prompt.
        norm: The normalized prompt.
        action: The action taken.
        reason: The reason for the action.
        log_path: The path to the log file.
        logger: The logger instance.
    """
    try:
        log_data = {"timestamp": ts,"request_id": request_id,"action": action,"reason": reason,"prompt_hash": hashlib.sha256((original or '').encode()).hexdigest(),"normalized_hash": hashlib.sha256((norm or '').encode()).hexdigest()}
        logger.info(json.dumps(log_data))
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data) + "\n")
    except Exception as e:
        logger.error(f"Log fail: {e}")
