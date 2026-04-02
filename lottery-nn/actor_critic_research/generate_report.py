"""
Generate research report from deep-research JSON results.
TOC fields: relevance_score, integration_point
"""

import json
import os
import re
import yaml
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
FIELDS_YAML  = Path(__file__).parent / "fields.yaml"
OUTPUT_MD    = Path(__file__).parent / "report.md"

TOC_FIELDS   = ["relevance_score", "integration_point"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s.strip())
    return s


def is_uncertain(value, field_name: str, uncertain_list: list) -> bool:
    if field_name in uncertain_list:
        return True
    if value is None or value == "":
        return True
    if isinstance(value, str) and "[uncertain]" in value:
        return True
    return False


def format_value(value) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        if all(isinstance(v, dict) for v in value):
            lines = []
            for item in value:
                parts = [f"{k}: {v}" for k, v in item.items()]
                lines.append(" | ".join(parts))
            return "<br>".join(lines)
        joined = ", ".join(str(v) for v in value)
        if len(joined) > 120:
            return "<br>".join(str(v) for v in value)
        return joined
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            parts.append(f"**{k}**: {v}")
        return "; ".join(parts)
    text = str(value)
    if len(text) > 120:
        # wrap long text in blockquote-friendly format
        return text
    return text


def load_fields(fields_path: Path):
    with open(fields_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    fields = data.get("fields", [])
    categories = {}
    for field in fields:
        cat = field.get("description", "")
        # derive category from comment grouping in the yaml
        categories[field["name"]] = field
    return fields


def get_field_categories(fields_path: Path):
    """Parse fields.yaml preserving category order."""
    with open(fields_path, encoding="utf-8") as f:
        content = f.read()
    data = yaml.safe_load(content)
    fields = data.get("fields", [])

    # Re-parse to extract comment categories
    category_order = []
    current_cat = "General"
    cat_fields = {}

    lines = content.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# ---"):
            # category header follows
            pass
        elif stripped.startswith("# ") and not stripped.startswith("# -"):
            current_cat = stripped[2:].strip(" -")
        elif stripped.startswith("- name:"):
            fname = stripped.split(":", 1)[1].strip()
            if current_cat not in cat_fields:
                cat_fields[current_cat] = []
                category_order.append(current_cat)
            cat_fields[current_cat].append(fname)

    return category_order, cat_fields


def load_all_results():
    items = []
    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            try:
                data = json.load(f)
                data["_source_file"] = json_file.name
                items.append(data)
            except json.JSONDecodeError as e:
                print(f"  WARNING: could not parse {json_file.name}: {e}")
    return items


def get_field(item: dict, field_name: str):
    """Look up a field in flat or nested JSON."""
    if field_name in item:
        return item[field_name]
    # search nested dicts
    for v in item.values():
        if isinstance(v, dict) and field_name in v:
            return v[field_name]
    return None


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(items, category_order, cat_fields):
    lines = []
    topic = "Actor-Critic Play Generation for Discrete Combinatorial Outputs"
    lines.append(f"# Research Report: {topic}\n")
    lines.append(f"**Items researched:** {len(items)}  ")
    lines.append(f"**Output directory:** `actor_critic_research/results/`\n")

    # Sort items by relevance_score descending
    def sort_key(item):
        score = get_field(item, "relevance_score")
        try:
            return -int(score)
        except (TypeError, ValueError):
            return 0

    items_sorted = sorted(items, key=sort_key)

    # -----------------------------------------------------------------------
    # TOC
    # -----------------------------------------------------------------------
    lines.append("## Table of Contents\n")
    lines.append("| # | Item | Relevance | Integration Point |")
    lines.append("|---|------|-----------|-------------------|")

    for idx, item in enumerate(items_sorted, 1):
        name = get_field(item, "name") or item.get("_source_file", "unknown")
        anchor = slugify(str(name))
        score = get_field(item, "relevance_score")
        intpt = get_field(item, "integration_point") or ""
        # shorten integration_point
        intpt_short = intpt.split("(")[0].strip() if intpt else ""
        stars = ("★" * int(score) if score and str(score).isdigit() else str(score))
        lines.append(f"| {idx} | [{name}](#{anchor}) | {stars} ({score}/5) | {intpt_short} |")

    lines.append("")

    # -----------------------------------------------------------------------
    # Detailed sections
    # -----------------------------------------------------------------------
    lines.append("---\n")
    lines.append("## Detailed Research Results\n")

    for item in items_sorted:
        uncertain_list = item.get("uncertain", [])
        name = get_field(item, "name") or item.get("_source_file", "unknown")
        anchor = slugify(str(name))
        item_type = get_field(item, "type") or ""
        score = get_field(item, "relevance_score")
        intpt = get_field(item, "integration_point") or ""

        lines.append(f"### {name} {{#{anchor}}}\n")
        lines.append(f"**Type:** {item_type} | **Relevance:** {score}/5 | **Integration:** {intpt}\n")

        # known fields per category
        known_fields = set()
        for cat in category_order:
            fields_in_cat = cat_fields.get(cat, [])
            if not fields_in_cat:
                continue

            cat_lines = []
            for fname in fields_in_cat:
                known_fields.add(fname)
                if fname in ("name", "type"):
                    continue  # already in header
                value = get_field(item, fname)
                if is_uncertain(value, fname, uncertain_list):
                    continue
                formatted = format_value(value)
                if not formatted:
                    continue
                cat_lines.append((fname, formatted))

            if cat_lines:
                lines.append(f"**{cat}**\n")
                for fname, formatted in cat_lines:
                    label = fname.replace("_", " ").title()
                    if len(formatted) > 200:
                        lines.append(f"- **{label}:**\n  > {formatted}\n")
                    else:
                        lines.append(f"- **{label}:** {formatted}")
                lines.append("")

        # extra fields not in fields.yaml
        skip_keys = {"_source_file", "uncertain"} | known_fields
        extra = [(k, v) for k, v in item.items()
                 if k not in skip_keys and not is_uncertain(v, k, uncertain_list)]
        if extra:
            lines.append("**Other Info**\n")
            for k, v in extra:
                label = k.replace("_", " ").title()
                formatted = format_value(v)
                if formatted:
                    lines.append(f"- **{label}:** {formatted}")
            lines.append("")

        # uncertain fields list
        if uncertain_list:
            lines.append("**Uncertain Fields**\n")
            for u in uncertain_list:
                lines.append(f"- {u}")
            lines.append("")

        lines.append("---\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading fields from {FIELDS_YAML}...")
    category_order, cat_fields = get_field_categories(FIELDS_YAML)
    print(f"  Categories: {category_order}")

    print(f"Loading results from {RESULTS_DIR}...")
    items = load_all_results()
    print(f"  Loaded {len(items)} items.")

    print("Generating report...")
    report = build_report(items, category_order, cat_fields)

    OUTPUT_MD.write_text(report, encoding="utf-8")
    print(f"Report saved to: {OUTPUT_MD}")
    print(f"  Size: {len(report):,} characters, {report.count(chr(10)):,} lines")
