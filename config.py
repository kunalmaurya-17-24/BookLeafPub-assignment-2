"""
config.py
=========
Single source of truth for all configuration.
- Loads secrets from .env via python-dotenv
- Defines all physical cover dimensions and safe-zone constants

IMPORTANT: All physical measurements are stored in MILLIMETRES.
The detection engine converts to pixels at runtime using the image DPI.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ---------------------------------------------------------------------------
# Google Cloud — Service Account
# ---------------------------------------------------------------------------
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "service_account.json")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")

# ---------------------------------------------------------------------------
# Gemini AI
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3-flash-preview"  # Current stable model as of 2026

# ---------------------------------------------------------------------------
# Airtable
# ---------------------------------------------------------------------------
AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY", "")
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID", "")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME", "CoverValidations")

# ---------------------------------------------------------------------------
# Email (Gmail SMTP with App Password)
# Notes:
#   - Service accounts cannot send Gmail on personal accounts.
#   - Use Gmail App Password: Google Account → Security → 2FA → App Passwords
#   - No sign-in needed by the evaluators — just set these env vars.
# ---------------------------------------------------------------------------
EMAIL_SENDER = os.getenv("EMAIL_SENDER", "")          # your-gmail@gmail.com
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")  # 16-char app password
SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", EMAIL_SENDER)

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))
PROCESSED_FILES_LOG = os.getenv("PROCESSED_FILES_LOG", "processed.json")
TEMP_DOWNLOAD_DIR = os.getenv("TEMP_DOWNLOAD_DIR", "temp_covers")

# ---------------------------------------------------------------------------
# Cover Physical Specifications (from the BookLeaf assignment)
# All values in MILLIMETRES
# ---------------------------------------------------------------------------

# Front cover: 5 inches × 8 inches
COVER_WIDTH_INCHES = 5.0
COVER_HEIGHT_INCHES = 8.0

# Conversion factor
MM_PER_INCH = 25.4

# Safe area margins (3mm on each side, 9mm from bottom)
SIDE_MARGIN_MM = 3.0          # left, right, top margins
BOTTOM_BADGE_ZONE_MM = 9.0    # reserved for "21st Century Emily Dickinson Award" emblem

# Detection buffer:
# We set this to 0.0 to enforce the strict 9mm rule.
BADGE_DETECTION_BUFFER_MM = 0.0

# Minimum acceptable image resolution
MIN_DPI = 300  # Below this triggers a "low resolution" flag

# ---------------------------------------------------------------------------
# Confidence thresholds for status classification
# ---------------------------------------------------------------------------
# If BOTH engines agree → 100 confidence
# If engines DISAGREE → REVIEW_NEEDED regardless of individual scores
CONFIDENCE_HIGH = 90     # Single-engine threshold to auto-PASS
CONFIDENCE_LOW = 50      # Below this always → REVIEW NEEDED

# ---------------------------------------------------------------------------
# ISBN author-email mapping
# ---------------------------------------------------------------------------
# In production this would be a proper database lookup.
# For the assignment demo, we use a hardcoded dictionary.
# Expand this with the ISBNs of your test covers.
ISBN_AUTHOR_MAP = {
    # Parisha Shodhan — Shabd
    "9789373147499": {"author_name": "Parisha Shodhan", "email": os.getenv("TEST_EMAIL_1", "")},
    "9798898652616": {"author_name": "Parisha Shodhan", "email": os.getenv("TEST_EMAIL_1", "")},
    # Pulak Das
    "9798898652753": {"author_name": "Pulak Das",       "email": os.getenv("TEST_EMAIL_2", "")},
    # Benny James SDB
    "9789373147994": {"author_name": "Benny James SDB", "email": os.getenv("TEST_EMAIL_3", "")},
    "9798898652364": {"author_name": "Benny James SDB", "email": os.getenv("TEST_EMAIL_3", "")},
    # Pratik Kolekar
    "9789373147765": {"author_name": "Pratik Kolekar",  "email": os.getenv("TEST_EMAIL_4", "")},
    # Additional ISBNs from assignment sample set
    "9789373148007": {"author_name": "Parisha Shodhan", "email": os.getenv("TEST_EMAIL_1", "")},
    "9789373147482": {"author_name": "Pulak Das",       "email": os.getenv("TEST_EMAIL_2", "")},
    "9789373148014": {"author_name": "Benny James SDB", "email": os.getenv("TEST_EMAIL_3", "")},
    "9789373147772": {"author_name": "Pratik Kolekar",  "email": os.getenv("TEST_EMAIL_4", "")},
    "9789373145068": {"author_name": "Ojal Jain", "email": os.getenv("TEST_EMAIL_5", "")},

}
