"""
airtable_logger.py
==================
Handles all read/write operations to the Airtable base.

Maps ValidationResult fields to specific Airtable columns:
- ISBN
- Author Name
- Status
- Confidence Score
- Issue Type
- Severity
- Correction Instructions
- Detection Timestamp
- Revision Count
"""

import logging
from datetime import datetime
from pyairtable import Table
import config
from schemas import ValidationResult

logger = logging.getLogger(__name__)

def log_validation_to_airtable(result: ValidationResult):
    """
    Creates or updates a record in Airtable for the given validation result.
    Aligns field names to the specific requirements of the assignment.
    """
    try:
        table = Table(config.AIRTABLE_API_KEY, config.AIRTABLE_BASE_ID, config.AIRTABLE_TABLE_NAME)
        
        # Prepare Issue strings
        issue_types = ", ".join(set(i.issue_type.value for i in result.issues)) if result.issues else "None"
        severities = ", ".join(set(i.severity.value for i in result.issues)) if result.issues else "None"
        corrections = "\n".join(f"- {i.correction}" for i in result.issues) if result.issues else "No corrections needed."

        fields = {
            "ISBN": result.isbn,
            "Author Name": result.author_name or "Unknown",
            "Status": result.status.value,
            "Confidence Score": result.confidence,
            "Issue Type": issue_types,
            "Severity": severities,
            "Correction Instructions": corrections,
            "Detection Timestamp": datetime.now().isoformat(),
            "Revision Count": 1  # Logic for tracking revisions could be added here
        }

        # Check if record already exists for this ISBN to update instead of create
        existing = table.all(formula=f"{{ISBN}} = '{result.isbn}'")
        
        if existing:
            record_id = existing[0]["id"]
            # Increment revision count
            fields["Revision Count"] = existing[0]["fields"].get("Revision Count", 0) + 1
            table.update(record_id, fields)
            logger.info(f"[Airtable] Updated record for ISBN={result.isbn} (Revision {fields['Revision Count']})")
            return record_id
        else:
            new_record = table.create(fields)
            logger.info(f"[Airtable] Created new record for ISBN={result.isbn}")
            return new_record["id"]

    except Exception as e:
        logger.error(f"[Airtable] Failed to log result: {e}")
        return None
