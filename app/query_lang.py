"""Multilingual search: the manuals are mostly in English (some in German / Spanish), the
questions come in Spanish, English or German. BM25 (60 % of the ranking) only matches
identical words, so 'boquillas' or 'Düsen' never finds 'Nozzle'. Before searching, the
query gets the English (and German / Spanish) words of the domain terms it contains:

    'zeige mir die Düsen für KA-2 HD'  ->  '... nozzle nozzles boquilla'

* expand_query()   free, deterministic glossary (always on; QUERY_TRANSLATE=off disables)
* llm_search_query()  optional: QUERY_TRANSLATE=llm asks the cheap agent model for English
                   search keywords when the question is not in English (+1 small call)

Add your own terms to GLOSSARY (keys are folded: lower case, no accents, ü -> ue).
"""
from __future__ import annotations

import re

from app.textutils import fold

# folded stem (prefix) in any language  ->  words added to the search
GLOSSARY: dict[str, str] = {
    # nozzle / needle / air cap
    "boquill": "nozzle nozzles", "duese": "nozzle nozzles", "dues": "nozzle nozzles",
    "buse": "nozzle nozzles", "ugell": "nozzle nozzles", "bico": "nozzle nozzles",
    "tobera": "nozzle nozzles", "nozzle": "boquilla duese",
    "aguja": "needle needles nozzle needle", "nadel": "needle needles nozzle needle",
    "aiguille": "needle needles", "agulha": "needle needles", 
    "needle": "aguja nadel",
    "tapa de aire": "air cap", "casquillo de aire": "air cap", "capuchon": "air cap",
    "cabezal de aire": "air cap", "luftkappe": "air cap", "luftduese": "air cap",
    "chapeau d air": "air cap", "cappello aria": "air cap", "air cap": "luftkappe",
    # parts / maintenance
    "repuesto": "spare part spare parts", "refaccion": "spare part spare parts",
    "ersatzteil": "spare part spare parts", "piece de rechange": "spare part",
    "ricambi": "spare part spare parts", "peca de reposicao": "spare part",
    "junta": "seal gasket o-ring", "sello": "seal seals", "empaque": "seal gasket",
    "dichtung": "seal gasket", "o ring": "o-ring seal",
    "guarnizion": "seal gasket", "vedacao": "seal",
    "resorte": "spring", "muelle": "spring", "feder": "spring", "ressort": "spring",
    "molla": "spring",
    "piston": "piston", "kolben": "piston", "embolo": "piston plunger",
    "husillo": "spindle", "spindel": "spindle", "tornillo": "screw", "schraube": "screw",
    "tuerca": "nut", "mutter": "nut", "carcasa": "housing", "cuerpo": "housing body",
    "gehaeuse": "housing", "manguera": "hose", "schlauch": "hose",
    "valvula": "valve valves", "ventil": "valve valves", "vanne": "valve", "valvola": "valve",
    "bomba": "pump", "pumpe": "pump", "pompe": "pump", "pompa": "pump",
    "mantenimiento": "maintenance", "wartung": "maintenance", "entretien": "maintenance",
    "manutenzione": "maintenance", "manutencao": "maintenance",
    "limpieza": "cleaning", "limpiar": "cleaning clean", "reinigung": "cleaning",
    "nettoyage": "cleaning", "pulizia": "cleaning",
    "instalacion": "installation install", "montaje": "assembly installation",
    "montage": "assembly installation", "desmontaje": "disassembly dismantling",
    "demontage": "disassembly dismantling",
    "falla": "malfunction fault", "averia": "malfunction fault", "stoerung": "malfunction fault",
    "fehler": "fault error", "panne": "malfunction",
    # properties
    "presion": "pressure", "druck": "pressure", "pression": "pressure", "pressione": "pressure",
    "caudal": "flow flow rate", "durchfluss": "flow flow rate", "debit": "flow",
    "portata": "flow", "vazao": "flow",
    "temperatura": "temperature", "temperatur": "temperature",
    "viscosidad": "viscosity", "viskositaet": "viscosity",
    "diametro": "diameter dia", "durchmesser": "diameter dia", "diametre": "diameter dia",
    "tamano": "size", "groesse": "size", "taille": "size", "dimensione": "size",
    "medida": "size dimension", "abmessung": "dimension",
    "peso": "weight", "gewicht": "weight", "poids": "weight",
    "acero inoxidable": "stainless steel", "inoxidable": "stainless steel",
    "edelstahl": "stainless steel", "inox": "stainless steel",
    "laton": "brass", "messing": "brass",
    "adhesiv": "adhesive", "pegamento": "adhesive glue", "kleb": "adhesive glue",
    "colle": "adhesive glue", "colla": "adhesive glue",
    "estandar": "standard",
    "version": "version", "ausfuehrung": "version", "variante": "version variant",
    "patron de rociado": "spray pattern", "abanico": "spray pattern flat",
    "spruehbild": "spray pattern", "rociado": "spray", "pulverizacion": "spray atomizing",
    "atomizad": "atomizing spray", "zerstaeub": "atomizing spray", "sprueh": "spray",
    "redond": "round", "rund": "round", "plano": "flat", "flach": "flat",
    "aire": "air", "luft": "air",
    "tabla": "table", "tabelle": "table", "tableau": "table", "tabella": "table",
    "dibujo": "drawing", "zeichnung": "drawing", "despiece": "exploded view parts list",
    "explosionszeichnung": "exploded view",
    "hoja tecnica": "datasheet technical data", "ficha tecnica": "datasheet technical data",
    "datenblatt": "datasheet technical data", "fiche technique": "datasheet",
    "sensor": "sensor", "detector": "sensor", "senal": "signal", "signal": "signal",
}
_KEYS = sorted(GLOSSARY, key=len, reverse=True)
_KEY_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(k.strip()) for k in _KEYS) + r")")

_EN_HINTS = {"the", "and", "of", "for", "show", "what", "which", "how", "please", "list",
             "available", "with", "is", "are", "me", "give", "tell"}
_ES_DE_HINTS = {"el", "la", "los", "las", "de", "del", "para", "que", "por", "una", "como",
                "cual", "cuales", "dame", "muestra", "muestrame", "die", "der", "das", "und",
                "fuer", "zeige", "mir", "bitte", "welche", "wie", "ist", "sind", "des", "le",
                "les", "quel", "quelle", "pour", "il", "gli", "per", "della"}


def expand_query(query: str) -> str:
    """query + the domain words of other languages it implies (deterministic, free)."""
    from app.config import settings
    if getattr(settings, "query_translate", "glossary") == "off":
        return query
    f = fold(query)
    extra: list[str] = []
    for m in _KEY_RE.finditer(f):
        key = next((k for k in _KEYS if k.strip() == m.group(1)), m.group(1))
        for w in GLOSSARY[key].split():
            if w not in extra and w not in f.split():
                extra.append(w)
    return f"{query} {' '.join(extra)}" if extra else query


def looks_english(query: str) -> bool:
    words = set(fold(query).split())
    return len(words & _EN_HINTS) >= len(words & _ES_DE_HINTS)


def llm_search_query(llm, meter, question: str) -> str:
    """English search keywords for a non-English question (QUERY_TRANSLATE=llm)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    sys = ("Rewrite the user's question as an English search query for technical manuals of "
           "dispensing valves, nozzles, pumps and spare parts. Keep model names, codes and "
           "numbers exactly as written. Output ONLY the query, max 20 words.")
    try:
        out = meter.invoke(llm, [SystemMessage(content=sys), HumanMessage(content=question)],
                           stage="translate")
        return " ".join(str(out).split())[:300]
    except Exception:
        return ""


def build_vocab(words) -> dict[tuple[str, int], list[str]]:
    """Index of the corpus vocabulary by (first letter, length) for fast typo matching."""
    idx: dict[tuple[str, int], list[str]] = {}
    for w in words:
        if isinstance(w, str) and 4 <= len(w) <= 30 and w.isalpha():
            idx.setdefault((w[0], len(w)), []).append(w)
    return idx


def spell_fix(query: str, vocab: set, idx: dict) -> str:
    """'nozles' is not in any document -> add the closest word that is ('nozzles').
    Only for words of 5+ letters missing from the index; numbers and codes untouched."""
    import difflib
    from app.textutils import simple_tokenize
    if not vocab:
        return query
    extra = []
    for t in dict.fromkeys(simple_tokenize(query)):
        if len(t) < 5 or not t.isalpha() or t in vocab:
            continue
        cands = [w for d in (-2, -1, 0, 1, 2) for w in idx.get((t[0], len(t) + d), [])]
        best = difflib.get_close_matches(t, cands, n=1, cutoff=0.83)
        if best and best[0] not in extra:
            extra.append(best[0])
    return f"{query} {' '.join(extra)}" if extra else query
