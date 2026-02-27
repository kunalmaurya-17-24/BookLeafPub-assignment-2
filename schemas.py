"""
schemas.py
==========
All Pydantic data models used across the pipeline.
Every function that passes data between files uses these models —
no raw dicts, no untested data shapes.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FileType(str, Enum):
    PDF = "pdf"
    PNG = "png"


class ValidationStatus(str, Enum):
    PASS = "PASS"
    REVIEW_NEEDED = "REVIEW NEEDED"


class IssueType(str, Enum):
    BADGE_OVERLAP = "Badge Overlap"
    MARGIN_VIOLATION = "Margin Violation"
    LOW_RESOLUTION = "Low Resolution"
    TEXT_MISALIGNED = "Text Misaligned"
    BACK_COVER_ALIGNMENT = "Back Cover Alignment"
    UNKNOWN = "Unknown"


class Severity(str, Enum):
    CRITICAL = "Critical"
    WARNING = "Warning"
    INFO = "Info"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Pixel-space rectangle for a detected text region."""
    x: int
    y: int
    width: int
    height: int
    text: Optional[str] = None  # The actual text content detected

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def get_overlap_height(self, other: "BoundingBox") -> int:
        """Returns the vertical penetration depth (overlap height) in pixels."""
        return max(0, min(self.y2, other.y2) - max(self.y, other.y))

    def overlaps_with(self, other: "BoundingBox", tolerance: int = 0) -> bool:
        """
        Returns True if this box intersects with another box by more than the tolerance.
        If the overlap width or height is less than or equal to tolerance, it returns False.
        """
        # Calculate intersection rectangle dimensions
        overlap_w = min(self.x2, other.x2) - max(self.x, other.x)
        overlap_h = self.get_overlap_height(other)

        # Only returns True if there is an intersection larger than the tolerance threshold
        return overlap_w > tolerance and overlap_h > tolerance


class Issue(BaseModel):
    """A single detected layout issue."""
    issue_type: IssueType
    severity: Severity
    description: str
    correction: str  # Specific step-by-step instruction for the author


class EngineResult(BaseModel):
    """Output from a single validation engine (Geometry OR Vision)."""
    engine_name: str
    passed: bool
    issues: List[Issue] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=100.0)
    max_overlap_mm: float = 0.0  # Penetration depth in mm
    needs_precision_check: bool = False  # If True, triggers geometry engine for measurement
    front_cover_x_start_px: float = 0.0  # Horizontal coordinate where front cover begins


class CoverFile(BaseModel):
    """Metadata about the uploaded cover file."""
    isbn: str
    file_type: FileType
    filename: str
    file_path: Optional[str] = None  # Populated after download
    author_name: Optional[str] = None
    author_email: Optional[str] = None
    file_id: Optional[str] = None


class ValidationResult(BaseModel):
    """Final consolidated result after both engines have run."""
    isbn: str
    author_name: Optional[str] = None
    author_email: Optional[str] = None
    status: ValidationStatus
    confidence: float = Field(ge=0.0, le=100.0)
    issues: List[Issue] = Field(default_factory=list)
    geometry_result: Optional[EngineResult] = None
    vision_result: Optional[EngineResult] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processed: bool = False

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def critical_issues(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    @property
    def formatted_issues(self) -> str:
        """Returns issues as email-ready ❌ bullet points."""
        if not self.issues:
            return "✅ No issues detected."
        lines = []
        for issue in self.issues:
            icon = "❌" if issue.severity == Severity.CRITICAL else "⚠️"
            lines.append(f"{icon} {issue.issue_type.value}: {issue.description}")
        return "\n".join(lines)

    @property
    def formatted_corrections(self) -> str:
        """Returns correction steps as a numbered list."""
        if not self.issues:
            return "No corrections needed."
        lines = []
        for i, issue in enumerate(self.issues, 1):
            lines.append(f"{i}. {issue.correction}")
        return "\n".join(lines)


class AirtableRecord(BaseModel):
    """Flat model matching Airtable table column names."""
    ISBN: str
    Author_Name: Optional[str] = None
    Status: str
    Confidence_Score: float
    Issue_Type: Optional[str] = None
    Severity: Optional[str] = None
    Correction_Instructions: Optional[str] = None
    Detection_Timestamp: str
    Revision_Count: int = 0
    Visual_Annotations_URL: Optional[str] = None
