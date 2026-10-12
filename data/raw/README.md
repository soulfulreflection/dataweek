# Raw data (not committed)

Copy the challenge files here, keeping their original names:

| File | Used by the pipeline |
|---|---|
| `01_Introduction.xlsx` | yes – uncertainty parameters for cases 1–100 |
| `02_Field_Production.xlsx` | yes – observed history + simulated history of cases 1–100 |
| `03_Train_Cases_1.xlsx` … `08_Test_Cases.xlsx` | optional – grid properties, summarised by `tools/extract_grid_features.py` |
| `09_Template_Deliverable.xlsx` | yes – template that is filled with the 10 forecast cases |

The pipeline finds files by keyword (`Introduction`, `Field_Production`, `Template`), so a
prefix or a space instead of an underscore is fine.
