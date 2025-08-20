from typing import List, Dict
import re

def basic_quality_checks(questions: List[Dict]) -> Dict:
    issues = {"empty_questions": 0, "near_duplicates": 0}
    seen = set()
    for q in questions:
        text = q.get("question") or q.get("statement") or ""
        if len(text.strip()) < 5:
            issues["empty_questions"] += 1
        key = re.sub(r"\W+", "", text.lower())
        if key in seen:
            issues["near_duplicates"] += 1
        seen.add(key)
    return issues
