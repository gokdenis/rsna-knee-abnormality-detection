"""Size words for Baker's cyst and joint effusion in the training reports (multilingual keyword rules).

The host's definitions count only moderate / large effusions and Baker's cysts as positive, while self-distilled
targets call many report-small findings positive. For each study and finding this returns:
    "small"  - the finding is mentioned only with small / minimal / trace size words
    "large"  - the finding is mentioned with moderate / large size words
    "absent" - every mention is negated
    ""       - not mentioned, or no clear size
Usage: python -I scripts/report_severity.py data/raw/train.csv out.csv
"""

import re
import sys

import pandas as pd

TERMS = {
    "Baker's": r"baker|popl[ií]te\w*\s+(cyst|kist|zyste|kyste|quiste|cisto)|(cyst|kist|zyste|kyste|quiste|cisto)\w*\s+(de\s+)?popl[ií]te",
    "Effusion": r"effusion|erguss|[ée]panchement|derrame|efüzyon|efuzyon|eklem\s+s[ıi]v[ıi]s[ıi]|joint\s+fluid|hydrops|intraartikul\w*\s+fl[uü]ssigkeit",
}
SMALL = (r"\b(small|smaller|minimal|minor|trace|tiny|slight|mild|little|discrete|thin|physiologic\w*|"
         r"pequeñ[oa]|peque[nñ][oa]|m[ií]nim[oa]|leve|escas[oa]|discret[oa]|"
         r"klein\w*|gering\w*|wenig|diskret\w*|schmal\w*|"
         r"petit\w*|discr[eè]t\w*|faible|minime|"
         r"küçük|kucuk|minimal|az\s+miktar|hafif|eser|sınırlı)\b")
LARGE = (r"\b(moderate|large|big|marked|significant|considerable|extensive|massive|substantial|prominent|"
         r"moderad[oa]|abundante|grande|importante|significativ[oa]|marcad[oa]|"
         r"mäßig\w*|massig\w*|deutlich\w*|groß\w*|gross\w*|ausgeprägt\w*|reichlich\w*|"
         r"mod[ée]r[ée]\w*|abondant\w*|important\w*|"
         r"orta|büyük|buyuk|belirgin|fazla|yoğun)\b")
NEG = (r"\b(no|not|without|negative\s+for|absence|absent|free\s+of|"
       r"sin|sem|n[aã]o|ausencia|aus[eê]ncia|"
       r"kein\w*|ohne|nicht|"
       r"pas\s+de|absence\s+de|sans|"
       r"yok|yoktur|izlenmedi|saptanmadı|görülmedi)\b")


def sentences(text):
    return [s for s in re.split(r"(?<=[.;:\n])\s+|\n", str(text).lower()) if s.strip()]


def classify(text, finding):
    small = large = pos = neg = 0
    for s in sentences(text):
        if not re.search(TERMS[finding], s):
            continue
        if re.search(NEG, s) and not re.search(SMALL + "|" + LARGE, s):
            neg += 1
            continue
        pos += 1
        small += bool(re.search(SMALL, s))
        large += bool(re.search(LARGE, s))
    if pos == 0:
        return "absent" if neg else ""
    if large:
        return "large"
    if small:
        return "small"
    return ""


def main():
    df = pd.read_csv(sys.argv[1])
    out = pd.DataFrame({"StudyInstanceUID": df["StudyInstanceUID"]})
    for f in TERMS:
        out[f] = [classify(t, f) for t in df["Report"].fillna("")]
    out.to_csv(sys.argv[2], index=False)
    for f in TERMS:
        print(f, out[f].replace("", "unclear").value_counts().to_dict())


if __name__ == "__main__":
    main()
