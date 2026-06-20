"""Gateway entry point.

Run from the repo root:  uvicorn app:app --host 0.0.0.0 --port 8000
(``src`` is on the path via the editable install / pyproject ``pythonpath``.)
"""
from dotenv import load_dotenv

load_dotenv()

from genie.interface.bootstrap import create_app  # noqa: E402

app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
