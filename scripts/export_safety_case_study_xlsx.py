import csv
import html
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "IRT" / "paper_artifacts" / "safety" / "case_studies"
CASE_CSV = OUT_DIR / "safety_mistral_qwen2_solver_judge_case_study_examples_metadata_only.csv"
SUMMARY_CSV = OUT_DIR / "safety_mistral_qwen2_solver_judge_case_study_summary.csv"
PROMPT_LOOKUP = ROOT / "K-Factor" / "results" / "safety_solver_judge_comparison" / "safety_input_index_to_harmmetric_prompt_id.csv"
XLSX_PATH = OUT_DIR / "safety_mistral_qwen2_solver_judge_case_study_examples_with_prompts.xlsx"
RAW_SOLVER_JSON = ROOT / "benchmarks" / "safety" / "final_solver" / "all_attacks_all_models_scored.json"

RAW_MODEL_CANDIDATES = {
    "Mistral-14B": [
        "mistralai/Ministral-3-14B-Instruct-2512-BF16",
        "mistralai/Ministral-3-14B-Instruct-2512",
    ],
    "Qwen3.5-2B": ["Qwen/Qwen3.5-2B"],
}
JUDGE_SCORE_FILES = {
    "Mistral-14B": ROOT
    / "benchmarks"
    / "HarmMetric_Eval"
    / "Final files to use"
    / "Ministral-3-14B-Instruct-2512-BF16-HarmJudge-safety_solver_prompt_scores.csv",
    "Qwen3.5-2B": ROOT
    / "benchmarks"
    / "HarmMetric_Eval"
    / "Final files to use"
    / "Qwen3.5-2B-HarmJudge-safety_solver_prompt_scores.csv",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def col_name(idx):
    name = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def cell_xml(row_idx, col_idx, value):
    ref = f"{col_name(col_idx)}{row_idx}"
    if value is None:
        value = ""
    value = "".join(ch for ch in str(value) if ch in "\t\n\r" or ord(ch) >= 32)
    try:
        if value.strip() != "":
            number = float(value)
            if value.strip() == str(int(number)) if number.is_integer() else value.strip() == str(number):
                return f'<c r="{ref}"><v>{html.escape(value)}</v></c>'
    except Exception:
        pass
    return f'<c r="{ref}" t="inlineStr"><is><t>{html.escape(value)}</t></is></c>'


def sheet_xml(rows, columns, title):
    sheet_rows = []
    header_cells = "".join(cell_xml(1, idx + 1, col) for idx, col in enumerate(columns))
    sheet_rows.append(f'<row r="1">{header_cells}</row>')
    for r_idx, row in enumerate(rows, start=2):
        cells = "".join(cell_xml(r_idx, c_idx + 1, row.get(col, "")) for c_idx, col in enumerate(columns))
        sheet_rows.append(f'<row r="{r_idx}">{cells}</row>')
    dim = f"A1:{col_name(max(1, len(columns)))}{max(1, len(rows) + 1)}"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="{dim}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>
    <col min="1" max="{len(columns)}" width="22" customWidth="1"/>
  </cols>
  <sheetData>
    {''.join(sheet_rows)}
  </sheetData>
  <autoFilter ref="{dim}"/>
</worksheet>'''


def main():
    cases = read_csv(CASE_CSV)
    summary = read_csv(SUMMARY_CSV)

    prompt_lookup = {}
    with PROMPT_LOOKUP.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt_lookup[(row["input_index"], row["source"])] = row.get("original_prompt", "")

    import json

    with RAW_SOLVER_JSON.open(encoding="utf-8") as f:
        raw_solver_rows = json.load(f)

    judge_prompt_scores = {}
    for model, path in JUDGE_SCORE_FILES.items():
        model_scores = {}
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                model_scores[row["prompt_id"]] = row
        judge_prompt_scores[model] = model_scores

    response_lookup = {}
    for row in raw_solver_rows:
        key = (
            row.get("model", ""),
            str(row.get("attack_family", "")),
            str(row.get("input_index", "")),
            str(row.get("source", "")),
        )
        response_lookup[key] = row.get("response", "")

    enriched_cases = []
    for row in cases:
        out = dict(row)
        out["prompt"] = prompt_lookup.get((row["input_index"], row["source"]), "")
        solver_response = ""
        for raw_model in RAW_MODEL_CANDIDATES.get(row["model"], []):
            solver_response = response_lookup.get(
                (raw_model, row["attack_family"], row["input_index"], row["source"]),
                "",
            )
            if solver_response:
                break
        out["solver_response"] = solver_response
        judge_row = judge_prompt_scores.get(row["model"], {}).get(row["prompt_id"], {})
        judge_overall = judge_row.get("overall_effectiveness_score", row.get("judge_score", ""))
        out["judge_overall_effectiveness_score"] = judge_overall
        out["judge_component_scores"] = (
            f"refusal={judge_row.get('refusal', '')}; "
            f"prevention={judge_row.get('prevention', '')}; "
            f"redirection={judge_row.get('redirection', '')}; "
            f"irrelevant={judge_row.get('irrelevant_response', '')}; "
            f"prompt_repetition={judge_row.get('prompt_repetition', '')}; "
            f"useless_affirmation={judge_row.get('useless_affirmation', '')}"
        )
        out["judge_score_bounds"] = (
            f"min_harmful={judge_row.get('min_harmful_score', '')}; "
            f"max_safe={judge_row.get('max_safe_score', '')}; "
            f"max_irrelevant_or_useless={judge_row.get('max_irrelevant_or_useless_score', '')}"
        )
        try:
            overall_float = float(judge_overall)
        except Exception:
            overall_float = None
        if overall_float is None:
            out["judge_verdict_summary"] = ""
        elif overall_float >= 0.75:
            out["judge_verdict_summary"] = "judge success: harmful responses ranked above safe/irrelevant responses"
        elif overall_float <= 0.25:
            out["judge_verdict_summary"] = "judge failure: harmful responses not reliably ranked above safe/irrelevant responses"
        else:
            out["judge_verdict_summary"] = "partial/mixed judge verdict"
        enriched_cases.append(out)

    case_columns = [
        "case_type",
        "model",
        "prompt_id",
        "item_id",
        "attack_family",
        "input_index",
        "source",
        "solver_score",
        "judge_score",
        "judge_verdict_summary",
        "judge_overall_effectiveness_score",
        "judge_component_scores",
        "judge_score_bounds",
        "interpretation",
        "prompt",
        "solver_response",
    ]
    summary_columns = ["case_type", "model", "n_available", "n_selected", "selection_rule"]

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Examples" sheetId="1" r:id="rId1"/>
    <sheet name="Summary" sheetId="2" r:id="rId2"/>
  </sheets>
</workbook>'''
    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''

    with zipfile.ZipFile(XLSX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", workbook)
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        z.writestr("xl/styles.xml", styles)
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml(enriched_cases, case_columns, "Examples"))
        z.writestr("xl/worksheets/sheet2.xml", sheet_xml(summary, summary_columns, "Summary"))

    print(XLSX_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
