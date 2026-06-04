# scripts/

Utility scripts for generating case study artifacts from safety benchmark results.

| Script | Purpose |
|---|---|
| `create_safety_case_study_notebook.py` | Selects representative safety case studies (items where solver and judge diverge) and writes them as a notebook to `IRT/charts_and_tables/safety/case_studies/` |
| `export_safety_case_study_xlsx.py` | Exports the safety case studies to an Excel spreadsheet for manual review |

## Usage

```bash
python scripts/create_safety_case_study_notebook.py
python scripts/export_safety_case_study_xlsx.py
```

## Outputs

- `IRT/charts_and_tables/safety/case_studies/` — generated notebooks and CSV tables, one per case study target model pair
