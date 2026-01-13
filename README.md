# Dental Classification — Streamlit Frontend

This Streamlit application provides a simple, professional UI to perform inference with the Ultralytics YOLO model trained in this repository.

Files added:
- `streamlit_app.py` — main Streamlit app (upload image, run inference, download result)
- `requirements.txt` — Python packages required to run the app

Quick start (PowerShell):

```powershell
cd "c:\Anas\Anas Working\Dental_classification"
python -m pip install -r application/requirements.txt
streamlit run application/streamlit_app.py
```

Notes:
- The app expects the trained model file at `training_pipeline/yolo11n.pt`. If you are training a different model, update the path in `streamlit_app.py` or replace the model file.
- The app uses the `ultralytics` API to load the YOLO model. Depending on your environment you may need a matching CUDA / torch installation.

If you want, I can:
- Add a small `utils.py` and unit tests
- Help configure a lightweight virtual environment for running the app
- Add deployment instructions (Docker, Heroku, Streamlit Cloud)
