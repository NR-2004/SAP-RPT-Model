# SAP-RPT-Model

PO delay prediction with SAP-RPT-1 on SAP AI Core (via BTP destination) + SAP S/4HANA OData.

1. `pip install -r requirements.txt`
2. `.env`: AI Core destination values + S/4HANA values; `destination_service_key.json` in this folder
3. `python main.py evaluate` (accuracy check on hold-out POs)
4. `python main.py` -> POST `/predict-delay`
