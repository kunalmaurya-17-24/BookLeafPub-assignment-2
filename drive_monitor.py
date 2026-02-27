"""
drive_monitor.py
================
Watches a Google Drive folder for new book cover uploads and downloads them.

Authentication: Google Service Account (JSON key file)
  — No OAuth browser pop-up required.
  — Share your target Drive folder with the service account email before running.

Key functions:
  authenticate_drive()          → builds the authenticated Drive API client
  list_new_files(service)       → returns unprocessed files in the folder
  download_file(service, ...)   → saves a Drive file to local disk
  parse_isbn_from_filename()    → extracts ISBN from "1234567890123_text.pdf"
  mark_as_processed(isbn)       → records processed ISBNs to prevent re-runs
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

import config

logger = logging.getLogger(__name__)

# Google APIs that the service account needs access to
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def authenticate_drive():
    """
    Builds and returns an authenticated Google Drive API v3 service client.

    Reads the service account JSON key from the path set in GOOGLE_SERVICE_ACCOUNT_JSON.
    Make sure the service account has been granted 'Viewer' or 'Editor' access
    to the specific Google Drive folder you are monitoring.
    """
    credentials = service_account.Credentials.from_service_account_file(
        config.GOOGLE_SERVICE_ACCOUNT_JSON,
        scopes=SCOPES,
    )
    service = build("drive", "v3", credentials=credentials)
    logger.info("Google Drive service authenticated via service account.")
    return service


# ---------------------------------------------------------------------------
# File tracking (processed ISBNs are stored in a local JSON log file)
# ---------------------------------------------------------------------------

def _load_processed_set() -> set[str]:
    """Loads the set of already-processed Drive file IDs from disk."""
    log_path = Path(config.PROCESSED_FILES_LOG)
    if not log_path.exists():
        return set()
    try:
        with open(log_path, "r") as f:
            data = json.load(f)
        return set(data.get("processed_file_ids", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def _save_processed_set(processed: set[str]) -> None:
    """Persists the processed file IDs set to disk."""
    log_path = Path(config.PROCESSED_FILES_LOG)
    with open(log_path, "w") as f:
        json.dump({"processed_file_ids": list(processed)}, f, indent=2)


def mark_as_processed(file_id: str) -> None:
    """Adds a Drive file ID to the processed log so it won't be downloaded again."""
    processed = _load_processed_set()
    processed.add(file_id)
    _save_processed_set(processed)
    logger.debug(f"Marked as processed: file_id={file_id}")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def list_new_files(service) -> list[dict]:
    """
    Lists files in the monitored Google Drive folder that:
      - Are PDF or PNG
      - Have NOT been processed yet (not in the local processed log)

    Returns a list of Drive file metadata dicts with 'id', 'name', 'mimeType'.
    """
    processed = _load_processed_set()

    query = (
        f"'{config.GOOGLE_DRIVE_FOLDER_ID}' in parents "
        f"and (mimeType='application/pdf' or mimeType='image/png') "
        f"and trashed = false"
    )

    response = service.files().list(
        q=query,
        fields="files(id, name, mimeType, createdTime)",
        orderBy="createdTime desc",
        pageSize=50,
    ).execute()

    all_files = response.get("files", [])
    new_files = [f for f in all_files if f["id"] not in processed]

    logger.info(
        f"Drive folder scan: {len(all_files)} total files, {len(new_files)} new to process."
    )
    return new_files


# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------

def download_file(service, file_id: str, destination_path: str) -> str:
    """
    Downloads a single file from Google Drive to the local filesystem.

    Args:
        service:          Authenticated Drive service.
        file_id:          The Drive file's unique ID.
        destination_path: Full path where the file should be saved.

    Returns:
        The destination_path string on success.
    """
    os.makedirs(Path(destination_path).parent, exist_ok=True)

    request = service.files().get_media(fileId=file_id)
    with open(destination_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    logger.info(f"Downloaded file_id={file_id} → {destination_path}")
    return destination_path


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_isbn_from_filename(filename: str) -> str:
    """
    Extracts the ISBN from the BookLeaf naming convention:
      "1234567890123_text.pdf"  →  "1234567890123"
      "9789373147499_cover.png" →  "9789373147499"

    Raises ValueError if the filename doesn't match the expected format.
    """
    stem = Path(filename).stem  # Strip extension
    parts = stem.split("_")
    if not parts:
        raise ValueError(f"Cannot parse ISBN from filename: '{filename}'")
    isbn = parts[0]
    if not isbn.isdigit():
        raise ValueError(
            f"Expected ISBN (digits only) before '_' in filename '{filename}', got '{isbn}'."
        )
    return isbn


# ---------------------------------------------------------------------------
# High-level Orchestrator Helpers
# ---------------------------------------------------------------------------

def scan_for_new_covers() -> list[CoverFile]:
    """
    High-level entry point for main.py polling loop.
    Returns a list of CoverFile schemas for any new files found.
    """
    from schemas import CoverFile, FileType
    
    service = authenticate_drive()
    files = list_new_files(service)
    
    results = []
    for f in files:
        try:
            isbn = parse_isbn_from_filename(f["name"])
            file_type = FileType.PDF if f["mimeType"] == "application/pdf" else FileType.PNG
            
            results.append(CoverFile(
                file_id=f["id"],
                filename=f["name"],
                isbn=isbn,
                file_type=file_type,
                author_name=config.ISBN_AUTHOR_MAP.get(isbn, {}).get("author_name"),
                author_email=config.ISBN_AUTHOR_MAP.get(isbn, {}).get("email")
            ))
        except ValueError as e:
            logger.warning(f"Skipping file '{f['name']}': {e}")
            
    return results

def download_cover(cover: CoverFile) -> str:
    """Downloads a CoverFile to the temp directory and returns local path."""
    service = authenticate_drive()
    ext = ".pdf" if cover.file_type.value == "pdf" else ".png"
    dest = str(Path(config.TEMP_DOWNLOAD_DIR) / f"{cover.isbn}{ext}")
    return download_file(service, cover.file_id, dest)
