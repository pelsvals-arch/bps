"""Streamlit page entrypoint.

Run with: streamlit run src/betdash/app/dashboard_page.py
"""

from betdash.app.dashboard import run_dashboard_cached


if __name__ == "__main__":
    run_dashboard_cached()
else:
    # Streamlit executes the file as a script; ensure render on import.
    run_dashboard_cached()
