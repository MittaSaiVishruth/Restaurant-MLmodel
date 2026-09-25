from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# =========================================================
# 17. High-precision aspect ontology and patterns
# =========================================================
ASPECTS = [
    "Food Quality",
    "Service",
    "Ambience",
    "Value for Money",
    "Hygiene",
    "Overall Dining Experience",
]

# The ontology deliberately includes restaurant-domain event language,
# morphological variants, and novel descriptive vocabulary.
ASPECT_TRIGGERS = {
    "Food Quality": [
        "food", "food quality", "dish", "dishes", "meal", "cuisine", "taste", "tasty",
        "delicious", "flavour", "flavor", "flavourful", "flavorful", "tasteless",
        "bland", "stale", "freshly prepared", "cooking", "cooked", "slow-cooked",
        "slow cooked", "slow-cooking", "slow cooking", "undercooked", "overcooked",
        "over done", "overdone", "dry", "raw", "tough", "burnt", "oily", "salty",
        "spicy", "starter", "starters", "main course", "dessert", "desserts",
        "biryani", "kebab", "kebabs", "pulao", "pasta", "pizza", "rasam", "curry",
        "curries", "stew", "fish", "prawn", "prawns", "mutton", "chicken", "rice",
        "bread", "breads", "appam", "seafood", "portion", "portions", "quantity",
        "authentic", "quality", "show stopper", "show-stopper", "lip smacking", "below average", "nothing to brag about", "not up to mark", "not upto mark", "banging", "out of this world", "most authentic",
    ],
    "Service": [
        "service", "services", "hospitality", "staff", "server", "servers", "waiter",
        "waiters", "waitress", "manager", "attentive", "courteous", "polite",
        "helpful", "prompt", "quick service", "slow service", "poor service",
        "bad service", "waiting", "waited", "waiting time", "delayed", "delay",
        "jumbled order", "order was jumbled", "order was delayed", "no one attended",
        "no one really attended", "unattended", "attended to the table", "rushed",
        "disorganized", "disorganised", "response", "served", "serving",
        "arrived late", "arrived 10 mins late", "finger bowls", "service timing",
        "service team", "guided us through the menu", "knowledgeable", "fine dining etiquette",
        "above and beyond", "well-trained", "well trained", "no one attended", "no one attended to the table", "order jumbled", "order jumbled up", "follow ups", "multiple follow-ups",
    ],
    "Ambience": [
        "ambience", "ambiance", "atmosphere", "decor", "decoration", "nawabi decor",
        "royal ambiance", "royal ambience", "regal ambiance", "regal ambience",
        "majestic ambiance", "majestic ambience", "royal vibe", "beautiful royal vibe",
        "lush", "green", "peaceful", "elegant", "inviting", "garden", "garden-like",
        "garden like", "interior", "interior design", "lighting", "music", "seating",
        "dining hall", "dining room", "surroundings", "view", "architecture",
        "aesthetic", "aesthetic arrangements", "vibe", "cozy", "cosy", "quiet",
        "spacious", "crowded", "cramped", "intimate", "relaxing", "romantic",
        "well-maintained", "well maintained", "hidden away", "hustle bustle",
        "scale of the dining room", "lush", "green", "peaceful", "welcoming", "calming", "soothing", "unmatched", "special atmosphere",
    ],
    "Value for Money": [
        "value for money", "value", "price", "prices", "pricing", "price tag", "cost",
        "costs", "costing", "expensive", "overpriced", "over priced", "exorbitant",
        "exorbitant price", "exorbitant price tag", "steep", "steep price", "reasonable",
        "reasonable price", "reasonable pricing", "affordable", "cheap", "worth",
        "worth the price", "worth the money", "worth every rupee", "worth every single rupee",
        "money's worth", "money’s worth", "not worth", "poor value", "low on value",
        "high value for money", "complete value for money", "good value for money",
        "portion", "portions", "quantity", "less quantity", "inflated", "inflated prices",
        "higher side", "high side", "high price", "bill", "fortune", "burn a hole",
        "ridiculous price", "absurd price", "daylight looting", "bucks", "rupees",
        "₹", "₹2500", "₹ 2500", "rupees", "bucks", "pricing is", "prices are", "price is", "costing", "cost to quantity", "price to quality", "cost-to-quantity", "worth trying",
    ],
    "Hygiene": [
        "hygiene", "hygienic", "cleanliness", "sanitation", "sanitized", "sanitised",
        "clean", "spotless", "spotlessly clean", "pristine", "immaculate", "sparkling clean",
        "dust-free", "dust free", "flawlessly clean", "stringent hygiene",
        "hygiene standards", "hygiene practices", "food safety", "sanitized menus",
        "clean restrooms", "restrooms", "restroom", "washroom", "toilet", "table linens",
        "silverware", "hot towels", "safe", "safety", "polished", "dirty", "unclean",
        "unhygienic", "contaminated", "foul smell", "bad smell", "hair in food",
        "hair", "insect", "cockroach", "fly in", "dust", "dust-free",
    ],
    "Overall Dining Experience": [
        "overall dining experience", "dining experience", "overall experience", "experience",
        "dining at", "eating at", "visit", "visiting", "revisit", "revisiting",
        "go back", "come back", "look forward to going back", "recommend", "recommended",
        "highly recommend", "will recommend", "must try", "must visit", "satisfied",
        "delightful experience", "amazing experience", "memorable", "special occasion",
        "night to remember", "magical", "nothing short of magical", "10/10",
        "once-in-a-lifetime", "once in a lifetime", "royal feast", "feel like royalty",
        "feel like absolute royalty", "worth every rupee", "outstanding", "top-notch",
        "absolutely fantastic", "absolutely amazing", "walked out", "not coming back",
        "won't come back", "wouldn't come back", "disappointing experience",
    ],
}

ASPECT_POSITIVE_TERMS = {
    "Food Quality": {
        "good","great","excellent","amazing","awesome","delicious","tasty","wonderful",
        "perfect","fantastic","best","nice","rich","tender","soft","juicy","fresh",
        "flavourful","flavorful","satisfying","quality","exceptional","authentic",
        "banging","show","show-stopper","lip","smacking","perfection","out","world",
        "beautifully","well-presented","well","prepared","consistent","consistently",
        "delightful","refined",
    },
    "Service": {
        "good","great","excellent","amazing","awesome","wonderful","perfect","fantastic",
        "best","nice","friendly","courteous","prompt","quick","attentive","impeccable",
        "exceptional","exemplary","legendary","knowledgeable","warm","polite","helpful",
        "trained","genuinely","above","beyond","benchmark","hospitality","sincerely",
        "kind","cared","caring","efficient","refined",
    },
    "Ambience": {
        "good","great","excellent","amazing","awesome","wonderful","perfect","fantastic",
        "best","nice","pleasant","royal","regal","magnificent","majestic","spectacular",
        "beautiful","gorgeous","elegant","luxurious","relaxing","unmatched","quiet",
        "romantic","grand","special","aesthetic","lush","green","peaceful","inviting",
        "cozy","cosy","spacious","refined","modern","well-maintained","maintained",
        "hidden","calm","serene",
    },
    "Value for Money": {
        "good","great","excellent","reasonable","affordable","cheap","worth","worthwhile",
        "justified","fair","balances","balanced","complete","high","value","money",
        "quality","matching","decent",
    },
    "Hygiene": {
        "good","great","excellent","amazing","impeccable","spotless","pristine","immaculate",
        "sanitized","sanitised","clean","cleanliness","hygienic","sparkling","stringent",
        "safe","flawless","top-notch","top","polished","fresh","dust-free","maintained",
        "well-maintained",
    },
    "Overall Dining Experience": {
        "good","great","excellent","amazing","awesome","wonderful","perfect","fantastic",
        "best","nice","pleasant","magical","flawless","unparalleled","special","memorable",
        "magnificent","royal","gorgeous","relaxing","recommended","recommend","enjoyed",
        "enjoy","love","10","night","remember","quality","royalty","outstanding",
        "top-notch","delightful","satisfied","amazing","refined",
    },
}

ASPECT_NEGATIVE_TERMS = {
    "Food Quality": {
        "bad","worst","poor","terrible","horrible","pathetic","awful","stale","bland","oily",
        "salty","cold","disappointing","disappointed","tough","raw","burnt","tasteless",
        "small","overcooked","undercooked","dry","overdone","below","average","flavourless",
        "flavorless","shitty","weird","strange","lacked","nothing","brag","mark","disappointing",
        "not",
    },
    "Service": {
        "bad","worst","poor","terrible","horrible","pathetic","awful","slow","rude","cold",
        "disappointing","disappointed","frustrating","disorganized","disorganised","rushed",
        "less","ignored","late","waiting","delayed","delay","jumbled","unattended","unresponsive",
        "poorly","messy","forgot","forgotten",
    },
    "Ambience": {
        "bad","worst","poor","terrible","horrible","pathetic","awful","cramped","crowded",
        "uncomfortable","disappointing","less","rushed","noisy",
    },
    "Value for Money": {
        "bad","worst","poor","terrible","awful","expensive","overpriced","exorbitant","costly",
        "steep","small","less","low","disappointing","disappointed","dent","fortune","burn",
        "hole","inflated","ridiculous","absurd","strange","daylight","looting","not","high",
    },
    "Hygiene": {
        "bad","worst","poor","terrible","horrible","pathetic","awful","dirty","unclean",
        "unhygienic","contaminated","foul","smell","hair","insect","cockroach","dust",
        "unsafe",
    },
    "Overall Dining Experience": {
        "bad","worst","poor","terrible","horrible","pathetic","awful","disappointing",
        "disappointed","ruined","frustrating","not","never","avoid","walked","leaving",
        "notwithstanding", "not worth it", "waste of money", "waste of money", "avoid this place", "never again",
    },
}

# High-precision phrase overrides. Negative constructions always run first.
STRONG_NEGATIVE_PHRASES = [
    "not the best", "wasn't the best", "was not the best", "not a bad place", "not bad",
    "nothing to brag about", "below average", "not up to mark", "not upto mark", "not up-to-mark",
    "less than good", "less than great", "not really impressed", "not impressed",
    "not value for money", "not good value", "not great value", "not worth the money",
    "don't feel it offers good value", "dont feel it offers good value",
    "do not feel it offers good value", "doesn't offer good value", "doesnt offer good value",
    "does not offer good value", "does not translate to good value", "doesn't translate to good value",
    "didn't get my money's worth", "didnt get my money's worth",
    "wouldn't call this value for money", "wouldnt call this value for money",
    "poor value for money", "absolutely poor value for money", "terrible value",
    "terrible value for money", "extremely overpriced", "highly overpriced", "incredibly overpriced",
    "over priced", "steep price", "quite expensive", "too expensive", "high price", "price was high", "price is high", "prices were high", "prices are high",
    "higher side", "high side", "prices are so inflated", "price is very high",
    "burn a hole in your pocket", "burn a hole in your pockets", "massive dent",
    "very slow", "exceptionally slow", "extremely slow", "slow service", "poor service",
    "bad service", "rushed service", "disorganized service", "disorganised service",
    "order was jumbled", "order was delayed", "no one really attended", "no one attended",
    "not coming back", "won't come back", "wouldn't come back", "worst meal",
    "worst food", "worst service", "poor hygiene", "very unhygienic", "dirty washroom",
    "dirty toilet", "stale food", "do not recommend", "don't recommend", "won't recommend",
    "wouldn't recommend", "food is shitty", "food was below average", "not up to mark",
    "not upto mark", "food is nothing to brag about", "nothing to brag about",
    "daylight looting", "prices were absurd", "prices are absurd",
]

STRONG_NEUTRAL_PHRASES = [
    "not bad", "not a bad place", "nothing special", "decent enough", "okay", "ok",
    "average", "hard to say",
]

STRONG_POSITIVE_PHRASES = [
    "nothing short of magical", "10/10", "overall experience is a 10/10",
    "once-in-a-lifetime", "once in a lifetime", "night to remember", "absolute perfection",
    "worth every rupee", "worth every single rupee", "high value for money",
    "complete value for money", "good value for money", "flawless hygiene", "flawless experience",
    "impeccable service", "impeccable hygiene", "top-notch service", "top notch service",
    "top-notch hygiene", "top notch hygiene", "exceptional service", "service was impeccable",
    "service is impeccable", "service was exemplary", "above and beyond", "warm service",
    "deeply polite", "highly recommend", "strongly recommend", "must try", "must visit",
    "very good service", "excellent service", "excellent food", "great service", "great food",
    "clean and hygienic", "spotlessly clean", "sparkling clean", "pristine cleanliness",
    "immaculate", "sanitized", "sanitised", "sheer quality", "high quality", "rich nizami cuisine",
    "royal vibe", "regal ambiance", "regal ambience", "royal ambiance", "royal ambience",
    "majestic ambiance", "majestic ambience", "luxurious setting", "beautiful royal vibe",
    "highly relaxing", "well-trained", "well trained", "fair trade", "blown away",
    "out of this world", "incredibly good", "authentic and refined", "most authentic",
    "consistently delicious", "absolutely outstanding", "top-notch", "absolutely fantastic",
    "absolutely amazing", "lip smacking", "show stopper", "show-stopper", "truly delightful",
]

COOKING_SLOW_COMPOUNDS = {
    "slow cooked", "slow-cooked", "slow cooking", "slow-cooking",
    "slow roasted", "slow-roasted", "slow roasting", "slow-roasting",
    "slow braised", "slow-braised", "slow braising", "slow-braising",
}

CONTRAST_RE = re.compile(
    r"\s+(?:but|however|although|though|whereas|yet|while)\s+|;|\n+",
    flags=re.I,
)

def _morph_pattern(item):
    item = str(item).lower().strip()
    if item in {"₹"}:
        return r"₹\s*\d+(?:[,.]\d+)*"
    if " " not in item and "-" not in item and "'" not in item and "’" not in item:
        # Singular/plural tolerance for common ontology nouns.
        if item.isalpha() and len(item) > 3:
            return rf"\b{re.escape(item)}s?\b"
        return rf"\b{re.escape(item)}\b"
    return re.escape(item)

def _compile_pattern(items):
    return re.compile(
        "(?:" + "|".join(_morph_pattern(x) for x in sorted(items, key=len, reverse=True)) + ")",
        flags=re.I,
    )

ASPECT_TRIGGER_PATTERNS = {
    aspect: _compile_pattern(triggers)
    for aspect, triggers in ASPECT_TRIGGERS.items()
}
ASPECT_POSITIVE_PATTERNS = {
    aspect: _compile_pattern(terms)
    for aspect, terms in ASPECT_POSITIVE_TERMS.items()
}
ASPECT_NEGATIVE_PATTERNS = {
    aspect: _compile_pattern(terms)
    for aspect, terms in ASPECT_NEGATIVE_TERMS.items()
}
GLOBAL_NEGATIVE_PATTERN = _compile_pattern(STRONG_NEGATIVE_PHRASES)
GLOBAL_NEUTRAL_PATTERN = _compile_pattern(STRONG_NEUTRAL_PHRASES)
GLOBAL_POSITIVE_PATTERN = _compile_pattern(STRONG_POSITIVE_PHRASES)

POSITIVE_TERMS = sorted(set().union(*ASPECT_POSITIVE_TERMS.values()))
NEGATIVE_TERMS = sorted(set().union(*ASPECT_NEGATIVE_TERMS.values()))
NEGATION_WORDS = {
    "not","no","never","don't","dont","didn't","didnt","won't","wont",
    "cannot","can't","cant","hardly","wouldn't","wouldnt","doesn't","doesnt",
}

TRIGGER_LOOKUPS = {
    aspect: {str(item).lower(): str(item) for item in triggers}
    for aspect, triggers in ASPECT_TRIGGERS.items()
}


# =========================================================
# 18. Context extraction + recall-oriented aspect detection
# =========================================================
MAX_CONTEXT_WORDS = 30
MAX_RULE_NGRAM = 5

def normalize_text(text):
    text = str(text)
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_into_sentences(text):
    if pd.isna(text):
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]

def split_into_clauses(text):
    return [p.strip() for p in CONTRAST_RE.split(normalize_text(text)) if p.strip()]

def _pattern_matches(text, pattern):
    return list(pattern.finditer(normalize_text(text)))

def _matched_trigger_text(aspect, match):
    return match.group(0).strip()

def _context_around_match(clause, start, end, window_words=MAX_CONTEXT_WORDS):
    words = clause.split()
    if len(words) <= window_words:
        return normalize_text(clause)

    # Character-to-word mapping without cutting through the target trigger.
    char_ranges = []
    pos = 0
    for w in words:
        idx = clause.find(w, pos)
        char_ranges.append((idx, idx + len(w)))
        pos = idx + len(w)

    center = 0
    for i, (a, b) in enumerate(char_ranges):
        if a <= start <= b or a <= end <= b:
            center = i
            break

    half = window_words // 2
    left = max(0, center - half)
    right = min(len(words), left + window_words)
    left = max(0, right - window_words)
    return normalize_text(" ".join(words[left:right]))

def _contains_local_hygiene_context(text):
    t = normalize_text(text).lower()
    return bool(re.search(
        r"\b(?:hygiene|hygienic|clean|cleanliness|sanitation|sanitized|sanitised|"
        r"spotless|pristine|restroom|washroom|toilet|linen|silverware|dust|safety|"
        r"food safety)\b", t, flags=re.I
    ))

def _contains_service_behavior_context(text):
    t = normalize_text(text).lower()
    return bool(re.search(
        r"\b(?:helpful|courteous|polite|attentive|warm|friendly|prompt|quick|slow|"
        r"rude|late|delayed|delay|waited|waiting|served|serving|attended|unattended|"
        r"jumbled|disorganized|disorganised|responsive|unresponsive|manager|waiter|"
        r"server|hospitality|follow[- ]?up|order|service|staff)\b", t, flags=re.I
    ))

def _aspect_specific_evaluation_signal(context, aspect):
    t = normalize_text(context).lower()
    if aspect == "Food Quality":
        # Generic words such as `fresh` only count as food evidence when tied to food.
        if re.search(r"\b(?:food|dish|meal|taste|flavour|flavor|biryani|curry|fish|prawn|mutton|chicken|rice|dessert|starter|main course|bread|appam|rasam|soup|kebab|pizza|pasta)\b", t):
            return True
        return bool(ASPECT_POSITIVE_PATTERNS[aspect].search(t) or ASPECT_NEGATIVE_PATTERNS[aspect].search(t))
    if aspect == "Service":
        if _contains_local_hygiene_context(t) and not _contains_service_behavior_context(t):
            return False
        return _contains_service_behavior_context(t) or bool(ASPECT_POSITIVE_PATTERNS[aspect].search(t) or ASPECT_NEGATIVE_PATTERNS[aspect].search(t))
    if aspect == "Value for Money":
        return bool(ASPECT_POSITIVE_PATTERNS[aspect].search(t) or ASPECT_NEGATIVE_PATTERNS[aspect].search(t) or re.search(r"₹\s*\d|\b\d{2,5}\s*(?:bucks|rupees)\b", t))
    if aspect == "Hygiene":
        return _contains_local_hygiene_context(t)
    if aspect == "Ambience":
        return bool(ASPECT_POSITIVE_PATTERNS[aspect].search(t) or ASPECT_NEGATIVE_PATTERNS[aspect].search(t))
    return bool(ASPECT_POSITIVE_PATTERNS[aspect].search(t) or ASPECT_NEGATIVE_PATTERNS[aspect].search(t) or GLOBAL_POSITIVE_PATTERN.search(t) or GLOBAL_NEGATIVE_PATTERN.search(t))

def detect_aspect_evidence(text):
    """Detect all aspect candidates and retain the full local clause/window needed for evaluation."""
    text = normalize_text(text)
    if not text:
        return []

    records = []
    seen = set()

    for clause in split_into_clauses(text):
        for aspect in ASPECTS:
            matches = _pattern_matches(clause, ASPECT_TRIGGER_PATTERNS[aspect])
            for match in matches:
                trigger = _matched_trigger_text(aspect, match)
                context = _context_around_match(clause, match.start(), match.end())

                if not _aspect_specific_evaluation_signal(context, aspect):
                    continue

                key = (aspect, normalize_text(context).lower())
                if key in seen:
                    continue
                seen.add(key)

                records.append({
                    "Aspect": aspect,
                    "Trigger": trigger,
                    "Aspect_Context": context,
                    "Source_Clause": clause,
                    "Context_Word_Count": len(context.split()),
                })

    return records



# =========================================================
# 19. Aspect-aware sentiment + continuous evaluation score
# =========================================================
def _normalized_phrase(text):
    return normalize_text(text).lower()

def _phrase_word_count(phrase):
    return len(
        re.findall(
            r"\b[\w₹]+(?:[-'][\w₹]+)?\b",
            str(phrase),
        )
    )

def _phrase_strength(phrase):
    # Longer phrases receive more semantic weight than isolated words.
    n = _phrase_word_count(phrase)
    return {
        1: 0.55,
        2: 0.70,
        3: 0.82,
        4: 0.92,
    }.get(n, 1.0)

def _phrase_quality_score(sentiment, n_words, strong=False):
    """
    Convert a phrase-level sentiment signal into a 0-100 evaluation score.
    The score is deliberately more expressive than a simple 50/100 neutral
    midpoint so that multiple high-quality positive reviews do not collapse
    toward 50 merely because the sample is small.
    """
    n_words = max(1, int(n_words))

    if strong:
        positive_score = min(100.0, 88.0 + 4.0 * max(0, n_words - 2))
    else:
        positive_score = min(96.0, 76.0 + 6.0 * max(0, n_words - 1))

    if sentiment == "Positive":
        return positive_score

    if strong:
        negative_score = max(0.0, 12.0 - 4.0 * max(0, n_words - 2))
    else:
        negative_score = max(4.0, 24.0 - 6.0 * max(0, n_words - 1))

    return negative_score

def _best_phrase_match(text, phrases):
    t = _normalized_phrase(text)
    matches = []

    for phrase in phrases:
        p = _normalized_phrase(phrase)
        if not p:
            continue

        if p in t:
            matches.append(
                (
                    p,
                    _phrase_word_count(p),
                    _phrase_strength(p),
                )
            )

    if not matches:
        return None

    return max(
        matches,
        key=lambda x: (x[1], x[2], len(x[0])),
    )

def _class_confidence_evaluation_score(probabilities_row, classes):
    """
    Continuous sentiment-quality score from the classifier posterior.
    Positive confidence pushes upward from 50; negative confidence pushes
    downward; neutral remains near 50.
    """
    p = {
        str(c): float(probabilities_row[i])
        for i, c in enumerate(classes)
    }

    predicted_class = max(
        p,
        key=p.get,
    )

    confidence = max(p.values())

    if predicted_class == "Positive":
        return float(
            np.clip(
                50.0 + 50.0 * confidence,
                0.0,
                100.0,
            )
        )

    if predicted_class == "Negative":
        return float(
            np.clip(
                50.0 - 50.0 * confidence,
                0.0,
                100.0,
            )
        )

    return float(
        np.clip(
            50.0
            + 10.0
            * (
                p.get("Positive", 0.0)
                - p.get("Negative", 0.0)
            ),
            0.0,
            100.0,
        )
    )

def _combine_rule_and_ml_scores(
    rule_sentiment,
    rule_score,
    ml_sentiment,
    ml_score,
):
    """
    Preserve strong local evidence when the ML model agrees, while allowing
    the ML model to temper isolated rule matches when the signals conflict.
    """
    rule_score = float(rule_score)
    ml_score = float(ml_score)

    if rule_sentiment == ml_sentiment:
        if rule_sentiment == "Positive":
            return max(rule_score, ml_score)

        if rule_sentiment == "Negative":
            return min(rule_score, ml_score)

        return 0.5 * rule_score + 0.5 * ml_score

    return (
        0.70 * rule_score
        + 0.30 * ml_score
    )

def _explicit_override(context, aspect):
    text = _normalized_phrase(context)

    # Negative constructions are checked before generic terms such as
    # "average", "good", "best", etc.
    negative_patterns = [
        r"\b(?:not the best|wasn't the best|was not the best)\b",
        r"\b(?:nothing to brag about|below average|not up to mark|not upto mark|less than good|less than great)\b",
        r"\b(?:not|never|don't|dont|doesn't|doesnt|didn't|didnt|wouldn't|wouldnt)\b.{0,60}\b(?:good|great|best|worth|value|recommended|impressed|satisfied)\b",
        r"\b(?:not|never)\b.{0,35}\b(?:great|excellent|amazing|outstanding|perfect|exceptional)\b",
    ]

    if any(
        re.search(
            pattern,
            text,
            flags=re.I,
        )
        for pattern in negative_patterns
    ):
        return (
            "Negative",
            10.0,
            0.995,
            "Explicit negation/contrast rule",
        )

    if re.search(
        r"\b(?:not bad|not a bad place|nothing special|decent enough|okay|ok|average|hard to say)\b",
        text,
        flags=re.I,
    ):
        return (
            "Neutral",
            50.0,
            0.99,
            "Explicit neutral phrase",
        )

    strong_neg = _best_phrase_match(
        text,
        STRONG_NEGATIVE_PHRASES,
    )

    if strong_neg is not None:
        phrase, n, _ = strong_neg
        return (
            "Negative",
            _phrase_quality_score(
                "Negative",
                n,
                strong=True,
            ),
            0.99,
            f"Strong negative phrase ({n}-gram): {phrase}",
        )

    strong_pos = _best_phrase_match(
        text,
        STRONG_POSITIVE_PHRASES,
    )

    if strong_pos is not None:
        phrase, n, _ = strong_pos
        return (
            "Positive",
            _phrase_quality_score(
                "Positive",
                n,
                strong=True,
            ),
            0.99,
            f"Strong positive phrase ({n}-gram): {phrase}",
        )

    return (
        None,
        None,
        None,
        None,
    )

def _phrase_lexicon_score(context, aspect):
    text = _normalized_phrase(context)

    pos_candidates = (
        list(ASPECT_POSITIVE_TERMS[aspect])
        + list(STRONG_POSITIVE_PHRASES)
    )
    neg_candidates = (
        list(ASPECT_NEGATIVE_TERMS[aspect])
        + list(STRONG_NEGATIVE_PHRASES)
    )

    pos = _best_phrase_match(
        text,
        pos_candidates,
    )
    neg = _best_phrase_match(
        text,
        neg_candidates,
    )

    if pos is None and neg is None:
        return (
            None,
            None,
            None,
            None,
        )

    if pos is not None and (
        neg is None
        or (pos[1], pos[2], len(pos[0]))
        >= (neg[1], neg[2], len(neg[0]))
    ):
        phrase, n, _ = pos
        score = _phrase_quality_score(
            "Positive",
            n,
            strong=False,
        )

        return (
            "Positive",
            score,
            min(
                0.96,
                0.74 + 0.05 * n,
            ),
            f"5-gram-aware positive phrase ({n}-gram): {phrase}",
        )

    phrase, n, _ = neg
    score = _phrase_quality_score(
        "Negative",
        n,
        strong=False,
    )

    return (
        "Negative",
        score,
        min(
            0.96,
            0.74 + 0.05 * n,
        ),
        f"5-gram-aware negative phrase ({n}-gram): {phrase}",
    )

def _local_token_sentiment(context, aspect):
    protected = _normalized_phrase(
        _protect_cooking_phrases(context)
    )

    tokens = re.findall(
        r"[a-zA-Z_'-]+",
        protected,
    )

    pos_terms = ASPECT_POSITIVE_TERMS[aspect]
    neg_terms = ASPECT_NEGATIVE_TERMS[aspect]

    score = 0

    for i, tok in enumerate(tokens):
        sign = (
            1
            if tok in pos_terms
            else -1
            if tok in neg_terms
            else 0
        )

        if sign == 0:
            continue

        previous = tokens[
            max(0, i - 5):i
        ]

        if tok == "slow":
            continue

        if any(
            w in previous
            for w in NEGATION_WORDS
        ):
            sign *= -1

        score += sign

    if score >= 2:
        return (
            "Positive",
            78.0,
            0.85,
            "Local token evidence",
        )

    if score <= -2:
        return (
            "Negative",
            22.0,
            0.85,
            "Local token evidence",
        )

    return (
        None,
        None,
        None,
        None,
    )

def _protect_cooking_phrases(text):
    text = _normalized_phrase(text)

    for compound in sorted(
        COOKING_SLOW_COMPOUNDS,
        key=len,
        reverse=True,
    ):
        protected = (
            compound
            .replace(" ", "_")
            .replace("-", "_")
        )

        text = text.replace(
            compound,
            protected,
        )

    return text

def predict_context_sentiment_batch(
    contexts,
    vectorizer,
    model,
    aspects=None,
):
    contexts = [
        normalize_text(x)
        for x in contexts
    ]

    if aspects is None:
        aspects = [None] * len(contexts)

    if len(aspects) != len(contexts):
        raise ValueError(
            "aspects and contexts must have the same length"
        )

    X = vectorizer.transform(
        contexts
    )

    ml_pred = model.predict(X)

    probabilities = (
        model.predict_proba(X)
        if hasattr(model, "predict_proba")
        else None
    )

    classes = (
        list(model.classes_)
        if hasattr(model, "classes_")
        else [
            "Negative",
            "Neutral",
            "Positive",
        ]
    )

    ml_eval_scores = []

    for i in range(len(contexts)):
        if probabilities is not None:
            ml_eval_scores.append(
                _class_confidence_evaluation_score(
                    probabilities[i],
                    classes,
                )
            )
        else:
            ml_eval_scores.append(
                {
                    "Negative": 20.0,
                    "Neutral": 50.0,
                    "Positive": 80.0,
                }.get(
                    str(ml_pred[i]),
                    50.0,
                )
            )

    sentiments = []
    eval_scores = []
    confidences = []
    sources = []

    for i, context in enumerate(contexts):
        aspect = aspects[i]
        ml_sent = str(ml_pred[i])
        ml_score = float(
            ml_eval_scores[i]
        )

        if aspect:
            rule_sent, rule_score, rule_conf, rule_source = _explicit_override(
                context,
                aspect,
            )

            if rule_sent is not None:
                final_score = _combine_rule_and_ml_scores(
                    rule_sent,
                    rule_score,
                    ml_sent,
                    ml_score,
                )

                sentiments.append(
                    rule_sent
                    if rule_sent != "Neutral"
                    else (
                        "Neutral"
                        if 40.0 <= final_score <= 60.0
                        else (
                            "Positive"
                            if final_score > 60.0
                            else "Negative"
                        )
                    )
                )
                eval_scores.append(
                    float(
                        np.clip(
                            final_score,
                            0,
                            100,
                        )
                    )
                )
                confidences.append(
                    float(rule_conf)
                )
                sources.append(
                    rule_source
                )
                continue

            phrase_sent, phrase_score, phrase_conf, phrase_source = _phrase_lexicon_score(
                context,
                aspect,
            )

            if phrase_sent is not None:
                final_score = _combine_rule_and_ml_scores(
                    phrase_sent,
                    phrase_score,
                    ml_sent,
                    ml_score,
                )

                sentiments.append(
                    phrase_sent
                )
                eval_scores.append(
                    float(
                        np.clip(
                            final_score,
                            0,
                            100,
                        )
                    )
                )
                confidences.append(
                    float(
                        phrase_conf
                    )
                )
                sources.append(
                    phrase_source
                )
                continue

            lex_sent, lex_score, lex_conf, lex_source = _local_token_sentiment(
                context,
                aspect,
            )

            if lex_sent is not None:
                final_score = _combine_rule_and_ml_scores(
                    lex_sent,
                    lex_score,
                    ml_sent,
                    ml_score,
                )

                sentiments.append(
                    lex_sent
                )
                eval_scores.append(
                    float(
                        np.clip(
                            final_score,
                            0,
                            100,
                        )
                    )
                )
                confidences.append(
                    float(
                        lex_conf
                    )
                )
                sources.append(
                    lex_source
                )
                continue

        sentiments.append(
            ml_sent
        )

        eval_scores.append(
            ml_score
        )

        confidences.append(
            float(
                max(probabilities[i])
                if probabilities is not None
                else 0.5
            )
        )

        sources.append(
            "ML probability fallback"
        )

    return (
        sentiments,
        eval_scores,
        confidences,
        sources,
    )


# =========================================================
# 20. Batch aspect evidence builder with continuous Evaluation_Score
# =========================================================
def build_aspect_evidence_table(review_df, vectorizer, model, batch_size=1200):
    records = []

    for _, row in review_df.iterrows():
        review = normalize_text(row["Review_clean"])
        for sentence in split_into_sentences(review):
            for item in detect_aspect_evidence(sentence):
                records.append({
                    "Source_Row_ID": row["Source_Row_ID"],
                    "Restaurant": row["Restaurant"],
                    "Review": review,
                    "Sentence": sentence,
                    "Aspect": item["Aspect"],
                    "Trigger": item["Trigger"],
                    "Source_Clause": item["Source_Clause"],
                    "Aspect_Context": item["Aspect_Context"],
                    "Context_Word_Count": item.get("Context_Word_Count", len(item["Aspect_Context"].split())),
                    "Rating_numeric": row.get("Rating_numeric", np.nan),
                    "Review_Sentiment_Class": row.get("sentiment_class", ""),
                })

    expected_cols = [
        "Source_Row_ID","Restaurant","Review","Sentence","Aspect","Trigger",
        "Source_Clause","Aspect_Context","Context_Word_Count","Rating_numeric",
        "Review_Sentiment_Class","ML_Sentiment","ML_Confidence","ML_Evaluation_Score",
        "Aspect_Sentiment","Evaluation_Score","Sentiment_Confidence","Decision_Source",
        "Evidence_Weight",
    ]

    evidence = pd.DataFrame(records)
    if evidence.empty:
        return pd.DataFrame(columns=expected_cols)

    evidence = evidence.drop_duplicates(
        subset=["Source_Row_ID","Aspect","Aspect_Context"]
    ).reset_index(drop=True)

    # Cache prediction for each unique (aspect, context) instead of re-transforming
    # repeated phrases across the corpus. This substantially reduces inference cost.
    unique = evidence[["Aspect", "Aspect_Context"]].drop_duplicates().reset_index(drop=True)
    unique_rows = []

    for start in range(0, len(unique), batch_size):
        batch = unique.iloc[start:start + batch_size]
        contexts = batch["Aspect_Context"].tolist()
        aspects = batch["Aspect"].tolist()

        X = vectorizer.transform(contexts)
        ml_sentiment = model.predict(X)
        probabilities = model.predict_proba(X)
        classes = list(model.classes_)
        ml_eval_scores = [
            _class_confidence_evaluation_score(probabilities[i], classes)
            for i in range(len(batch))
        ]
        ml_confidence = probabilities.max(axis=1)

        final_sentiment, evaluation_scores, final_confidences, sources = predict_context_sentiment_batch(
            contexts, vectorizer, model, aspects=aspects
        )

        out = batch.copy()
        out["ML_Sentiment"] = ml_sentiment
        out["ML_Confidence"] = ml_confidence
        out["ML_Evaluation_Score"] = ml_eval_scores
        out["Aspect_Sentiment"] = final_sentiment
        out["Evaluation_Score"] = evaluation_scores
        out["Sentiment_Confidence"] = final_confidences
        out["Decision_Source"] = sources
        out["Evidence_Weight"] = 0.5 + 0.5 * out["Sentiment_Confidence"].astype(float)
        unique_rows.append(out)

    prediction_table = pd.concat(unique_rows, ignore_index=True)
    evidence = evidence.merge(
        prediction_table,
        on=["Aspect", "Aspect_Context"],
        how="left",
        validate="many_to_one",
    )

    return evidence[expected_cols]



# =========================================================
# 22. GastroEval scoring — review-level evidence + evaluation score
# =========================================================
ASPECT_WEIGHTS = {
    "Food Quality": 0.30,
    "Service": 0.20,
    "Ambience": 0.15,
    "Value for Money": 0.15,
    "Hygiene": 0.10,
    "Overall Dining Experience": 0.10,
}

CONFIDENCE_K = 5

def evidence_confidence(
    evidence_count,
    k=CONFIDENCE_K,
):
    if evidence_count <= 0:
        return 0.0

    return evidence_count / (
        evidence_count + k
    )

def collapse_review_aspect_evidence(
    evidence_df,
):
    """
    Collapse multiple clauses from the same review/aspect into ONE evidence unit.

    This prevents one long review from being counted four or five times simply
    because the same aspect was mentioned in several clauses.
    """
    if evidence_df.empty:
        return evidence_df.copy()

    rows = []

    group_cols = [
        "Source_Row_ID",
        "Restaurant",
        "Aspect",
    ]

    for (
        source_id,
        restaurant,
        aspect,
    ), group in evidence_df.groupby(
        group_cols,
        sort=False,
    ):
        weights = (
            0.5
            + 0.5
            * group[
                "Sentiment_Confidence"
            ].astype(float)
        ).to_numpy()

        scores = group[
            "Evaluation_Score"
        ].astype(float).to_numpy()

        combined_score = float(
            np.average(
                scores,
                weights=weights,
            )
        )

        if combined_score >= 60.0:
            combined_sentiment = "Positive"
        elif combined_score <= 40.0:
            combined_sentiment = "Negative"
        else:
            combined_sentiment = "Neutral"

        representative_idx = (
            np.abs(
                group[
                    "Evaluation_Score"
                ].astype(float)
                - 50.0
            ).idxmax()
        )

        representative = group.loc[
            representative_idx
        ]

        rows.append({
            "Source_Row_ID": source_id,
            "Restaurant": restaurant,
            "Aspect": aspect,
            "Review": representative.get(
                "Review",
                "",
            ),
            "Aspect_Context": " | ".join(
                dict.fromkeys(
                    group[
                        "Aspect_Context"
                    ].astype(str).tolist()
                )
            ),
            "Evaluation_Score": combined_score,
            "Aspect_Sentiment": combined_sentiment,
            "Sentiment_Confidence": float(
                np.average(
                    group[
                        "Sentiment_Confidence"
                    ].astype(float),
                    weights=np.ones(
                        len(group)
                    ),
                )
            ),
            "Evidence_Weight": float(
                np.mean(weights)
            ),
            "Context_Count": int(
                len(group)
            ),
            "Decision_Source": "Review-aspect aggregation",
        })

    return pd.DataFrame(rows)

def compute_aspect_quality_from_rows(
    rows,
):
    if rows.empty:
        return np.nan

    weights = (
        rows["Evidence_Weight"]
        .astype(float)
        .clip(lower=0.05)
    )

    scores = (
        rows["Evaluation_Score"]
        .astype(float)
        .clip(0, 100)
    )

    return float(
        np.average(
            scores,
            weights=weights,
        )
    )

def aggregate_aspect_evidence(
    evidence_df,
):
    scoring_evidence = (
        collapse_review_aspect_evidence(
            evidence_df
        )
    )

    base = pd.DataFrame({
        "Aspect": ASPECTS
    })

    if scoring_evidence.empty:
        base["Evidence_Count"] = 0
        base["Negative"] = 0
        base["Neutral"] = 0
        base["Positive"] = 0
        base["Aspect_Score"] = np.nan
        base["Evidence_Confidence"] = 0.0
        base["Evidence_Confidence_Pct"] = 0.0
        base["Mean_Evaluation_Score"] = np.nan
        return base

    rows = []

    for aspect in ASPECTS:
        subset = scoring_evidence[
            scoring_evidence["Aspect"]
            == aspect
        ]

        rows.append({
            "Aspect": aspect,
            "Evidence_Count": int(
                len(subset)
            ),
            "Negative": int(
                (
                    subset["Aspect_Sentiment"]
                    == "Negative"
                ).sum()
            ),
            "Neutral": int(
                (
                    subset["Aspect_Sentiment"]
                    == "Neutral"
                ).sum()
            ),
            "Positive": int(
                (
                    subset["Aspect_Sentiment"]
                    == "Positive"
                ).sum()
            ),
            "Aspect_Score": compute_aspect_quality_from_rows(
                subset
            ),
            "Evidence_Confidence": evidence_confidence(
                len(subset)
            ),
            "Mean_Evaluation_Score": (
                float(
                    subset[
                        "Evaluation_Score"
                    ].mean()
                )
                if len(subset)
                else np.nan
            ),
        })

    summary = pd.DataFrame(
        rows
    )

    summary[
        "Evidence_Confidence_Pct"
    ] = (
        summary[
            "Evidence_Confidence"
        ]
        * 100
    ).round(2)

    return summary

def score_aspect_summary(
    summary,
):
    scoring = summary.copy()

    scoring["Base_Weight"] = (
        scoring["Aspect"]
        .map(ASPECT_WEIGHTS)
    )

    scoring["Supported_Weight"] = np.where(
        scoring["Evidence_Count"] > 0,
        scoring["Base_Weight"],
        0.0,
    )

    total_supported_weight = (
        scoring["Supported_Weight"].sum()
    )

    if total_supported_weight <= 0:
        scoring["Normalized_Weight"] = 0.0
        scoring["Weighted_Contribution"] = 0.0
        final_score = 50.0
    else:
        scoring["Normalized_Weight"] = (
            scoring["Supported_Weight"]
            / total_supported_weight
        )

        scoring["Weighted_Contribution"] = (
            scoring["Aspect_Score"]
            .fillna(50.0)
            * scoring["Normalized_Weight"]
        )

        final_score = float(
            scoring[
                "Weighted_Contribution"
            ].sum()
        )

    total_evidence = int(
        scoring["Evidence_Count"].sum()
    )

    supported_aspects = int(
        (
            scoring["Evidence_Count"]
            > 0
        ).sum()
    )

    coverage_pct = round(
        supported_aspects
        / len(ASPECTS)
        * 100,
        2,
    )

    overall_confidence_pct = round(
        evidence_confidence(
            total_evidence,
            k=10,
        )
        * 100,
        2,
    )

    return (
        round(final_score, 2),
        scoring,
        coverage_pct,
        overall_confidence_pct,
    )

def recommendation_label(
    score,
    total_evidence=0,
):
    if total_evidence == 0:
        return "Insufficient Evidence"

    if score >= 85:
        return "Highly Recommended"

    if score >= 75:
        return "Recommended"

    if score >= 65:
        return "Generally Recommended"

    if score >= 50:
        return "Consider with Caution"

    return "Not Recommended"

def evidence_quality_label(
    total_evidence,
):
    if total_evidence == 0:
        return "Insufficient Evidence"

    if total_evidence < 5:
        return "Low Evidence"

    if total_evidence < 15:
        return "Moderate Evidence"

    return "Strong Evidence"



# =========================================================
# 23. Restaurant-level scoring
# =========================================================
def build_restaurant_scores(
    evidence_df,
    restaurants,
):
    scoring_evidence = (
        collapse_review_aspect_evidence(
            evidence_df
        )
    )

    rows = []

    for restaurant in restaurants:
        for aspect in ASPECTS:
            if scoring_evidence.empty:
                subset = pd.DataFrame()
            else:
                subset = scoring_evidence[
                    (scoring_evidence["Restaurant"] == restaurant)
                    & (scoring_evidence["Aspect"] == aspect)
                ]

            count = int(
                len(subset)
            )

            row = {
                "Restaurant": restaurant,
                "Aspect": aspect,
                "Evidence_Count": count,
                "Negative": (
                    int(
                        (
                            subset["Aspect_Sentiment"]
                            == "Negative"
                        ).sum()
                    )
                    if count
                    else 0
                ),
                "Neutral": (
                    int(
                        (
                            subset["Aspect_Sentiment"]
                            == "Neutral"
                        ).sum()
                    )
                    if count
                    else 0
                ),
                "Positive": (
                    int(
                        (
                            subset["Aspect_Sentiment"]
                            == "Positive"
                        ).sum()
                    )
                    if count
                    else 0
                ),
                "Aspect_Score": (
                    compute_aspect_quality_from_rows(
                        subset
                    )
                    if count
                    else np.nan
                ),
                "Mean_Evaluation_Score": (
                    float(
                        subset[
                            "Evaluation_Score"
                        ].mean()
                    )
                    if count
                    else np.nan
                ),
                "Evidence_Confidence": evidence_confidence(
                    count
                ),
            }

            rows.append(row)

    summary = pd.DataFrame(
        rows
    )

    summary[
        "Evidence_Confidence_Pct"
    ] = (
        summary[
            "Evidence_Confidence"
        ]
        * 100
    ).round(2)

    summary["Base_Weight"] = (
        summary["Aspect"]
        .map(ASPECT_WEIGHTS)
    )

    summary["Supported_Weight"] = np.where(
        summary["Evidence_Count"] > 0,
        summary["Base_Weight"],
        0.0,
    )

    total_weight = (
        summary.groupby(
            "Restaurant"
        )["Supported_Weight"]
        .transform("sum")
    )

    summary["Normalized_Weight"] = np.where(
        total_weight > 0,
        summary["Supported_Weight"]
        / total_weight,
        0.0,
    )

    summary["Weighted_Contribution"] = (
        summary["Aspect_Score"]
        .fillna(50.0)
        * summary["Normalized_Weight"]
    )

    restaurant_scores = (
        summary.groupby(
            "Restaurant"
        )
        .agg(
            GastroEval_Score=(
                "Weighted_Contribution",
                "sum",
            ),
            Total_Aspect_Evidence=(
                "Evidence_Count",
                "sum",
            ),
            Aspects_With_Evidence=(
                "Evidence_Count",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),
            Mean_Evidence_Confidence_Pct=(
                "Evidence_Confidence_Pct",
                "mean",
            ),
            Mean_Evaluation_Score=(
                "Mean_Evaluation_Score",
                "mean",
            ),
        )
        .reset_index()
    )

    restaurant_scores[
        "GastroEval_Score"
    ] = np.where(
        restaurant_scores[
            "Total_Aspect_Evidence"
        ] > 0,
        restaurant_scores[
            "GastroEval_Score"
        ],
        50.0,
    ).round(2)

    restaurant_scores[
        "Aspect_Coverage_Pct"
    ] = (
        restaurant_scores[
            "Aspects_With_Evidence"
        ]
        / len(ASPECTS)
        * 100
    ).round(2)

    restaurant_scores[
        "Evidence_Quality"
    ] = (
        restaurant_scores[
            "Total_Aspect_Evidence"
        ]
        .apply(
            evidence_quality_label
        )
    )

    restaurant_scores[
        "Recommendation"
    ] = restaurant_scores.apply(
        lambda r: recommendation_label(
            float(
                r["GastroEval_Score"]
            ),
            int(
                r["Total_Aspect_Evidence"]
            ),
        ),
        axis=1,
    )

    return (
        restaurant_scores,
        summary,
    )


ARTIFACT_DIR = Path(__file__).resolve().parent / "gastroeval_artifacts"


def load_inference_artifacts(artifact_dir: str | Path | None = None):
    base = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    vectorizer = joblib.load(base / "tfidf_vectorizer.joblib")
    model = joblib.load(base / "sentiment_model.joblib")
    aspect_weights = joblib.load(base / "aspect_weights.joblib")
    aspect_config = joblib.load(base / "aspect_inference_config.joblib")
    scoring_config = joblib.load(base / "scoring_config.joblib")
    return vectorizer, model, aspect_weights, aspect_config, scoring_config


def analyze_reviews(reviews, restaurant="Restaurant", artifact_dir=None):
    """Run the final notebook's aspect/evidence/scoring inference on user reviews."""
    reviews = [normalize_text(x) for x in reviews if normalize_text(x)]
    vectorizer, model, aspect_weights, aspect_config, scoring_config = load_inference_artifacts(artifact_dir)
    rows = []
    for i, review in enumerate(reviews, start=1):
        rows.append({
            "Source_Row_ID": i,
            "Restaurant": restaurant,
            "Review_clean": review,
            "Rating_numeric": np.nan,
            "sentiment_class": "",
        })
    review_df = pd.DataFrame(rows)
    evidence = build_aspect_evidence_table(review_df, vectorizer, model)
    scores, aspect_summary = build_restaurant_scores(evidence, [restaurant])
    restaurant_row = scores[scores["Restaurant"] == restaurant].iloc[0].to_dict()
    aspect_summary = aspect_summary[aspect_summary["Restaurant"] == restaurant].copy()
    return {
        "restaurant_score": float(restaurant_row["GastroEval_Score"]),
        "restaurant_row": restaurant_row,
        "aspect_summary": aspect_summary,
        "evidence": evidence,
        "recommendation": restaurant_row["Recommendation"],
        "evidence_quality": restaurant_row["Evidence_Quality"],
        "coverage_pct": float(restaurant_row["Aspect_Coverage_Pct"]),
        "evidence_count": int(restaurant_row["Total_Aspect_Evidence"]),
    }
