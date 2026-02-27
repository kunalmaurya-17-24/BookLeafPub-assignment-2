"""
main.py
=======
The orchestrator — the script you run to start the whole system.

Flow (runs in an infinite polling loop):
  1. [drive_monitor]  Checks Google Drive for new covers.
  2. [detection]      Runs dual-engine validation.
  3. [airtable_logger] Logs result to Airtable.
  4. [email_sender]   Sends email to author.

Headless Polling:
  The loop is wrapped in a robust try-except to ensure the system 
  doesn't stop on single-file errors or API timeouts.
"""

import logging
import sys
import time
from pathlib import Path

import config
import airtable_logger
import drive_monitor
import email_sender
from detection import process_book_cover

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bookleaf_validator.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

def validate_environment():
    """Checks for required secrets and folders."""
    missing = []
    if not config.GEMINI_API_KEY: missing.append("GEMINI_API_KEY")
    if not config.AIRTABLE_API_KEY: missing.append("AIRTABLE_API_KEY")
    if not config.EMAIL_APP_PASSWORD: missing.append("EMAIL_APP_PASSWORD")
    
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)
    
    Path(config.TEMP_DOWNLOAD_DIR).mkdir(exist_ok=True)
    logger.info("Environment validated ✅")

def main():
    validate_environment()
    
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  BookLeaf Cover Validation System — Starting")
    logger.info(f"  Polling every {config.POLL_INTERVAL_SECONDS}s")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    while True:
        try:
            # 1. Poll for new files
            new_files = drive_monitor.scan_for_new_covers()
            
            if not new_files:
                logger.info("No new covers found. Waiting...")
            else:
                logger.info(f"Found {len(new_files)} new cover(s) to process.")
                
                for cover in new_files:
                    try:
                        logger.info(f"━━━ Processing ISBN={cover.isbn} ━━━")
                        
                        # Download
                        local_path = drive_monitor.download_cover(cover)
                        
                        # Validate
                        result = process_book_cover(local_path, cover.isbn, cover)
                        
                        # Log to Airtable
                        airtable_logger.log_validation_to_airtable(result)
                        
                        # Email
                        email_sender.notify_author(result)
                        
                        # Mark as processed in local log
                        drive_monitor.mark_as_processed(cover.file_id)
                        
                        logger.info(f"Successfully processed ISBN={cover.isbn}")
                        
                        # Rate limit protection (staying below 20 RPM for Gemini Free Tier)
                        time.sleep(2)
                        
                    except Exception as file_err:
                        logger.error(f"Error processing file {cover.filename}: {file_err}")
                        continue # Move to next file
                        
        except Exception as loop_err:
            logger.error(f"CRITICAL ERROR in polling loop: {loop_err}")
            logger.info("System will attempt to recover in 60s...")
            
        time.sleep(config.POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("System stopped by user.")
        sys.exit(0)
