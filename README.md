# SAP-RPT-Model

PO delay prediction with SAP-RPT-1 on SAP AI Core (via BTP destination) + SAP S/4HANA OData.

1. `pip install -r requirements.txt`
2. `.env`: AI Core destination values + S/4HANA values; `destination_service_key.json` in this folder
3. Training data: `po_history/po_delay_training_s4.csv` (completed PO items from S/4).
   Another file can be used with `TRAINING_CSV=...` in `.env`.
4. `python main.py evaluate` (accuracy check on hold-out POs)
5. `python main.py` -> http://127.0.0.1:8000/docs -> POST `/predict-delay`
