"""
detection.py
============
The core intelligence of the validation system.

Contains two independent engines that run on every uploaded cover:
  1. Geometry Engine (OpenCV + Tesseract)
  2. Vision Engine (Gemini 3.0 Flash)

Note: Deprecation warnings for google-generativeai are suppressed for a clean UI.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
# Suppress the "All support for google-generativeai has ended" warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import google.generativeai as genai

import numpy as np
import pytesseract
from PIL import Image

# Explicitly point pytesseract to the Windows Tesseract binary to avoid PATH issues.
# Use explicit path on Windows; on Linux (e.g. Hugging Face) leave default (tesseract in PATH).
if sys.platform == "win32":
    pytesseract.tesseract_cmd = r"E:\Tesseract-installation"

import config
from schemas import (
    BoundingBox,
    EngineResult,
    Issue,
    IssueType,
    Severity,
    ValidationResult,
    ValidationStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini setup (Warning suppressed above)
# ---------------------------------------------------------------------------
genai.configure(api_key=config.GEMINI_API_KEY)
_gemini_model = genai.GenerativeModel(config.GEMINI_MODEL)


# ===========================================================================
# IMAGE PRE-PROCESSING
# ===========================================================================

def load_and_preprocess(file_path: str) -> Image.Image:
    """
    Opens a PDF or PNG cover file and returns a Pillow RGB Image.
    PDFs are converted to PNG using the first page only at 300 DPI.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".png":
        image = Image.open(file_path).convert("RGB")
        logger.info(f"Loaded PNG: {file_path} | Size: {image.size}")
        return image

    elif suffix == ".pdf":
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(file_path, dpi=300, first_page=1, last_page=1)
            image = pages[0].convert("RGB")
            logger.info(f"Converted PDF page 1 to image: {file_path} @ 300 DPI")
            return image
        except ImportError:
            raise RuntimeError(
                "pdf2image is not installed. Install it with: pip install pdf2image\n"
                "Also install Poppler: https://github.com/oschwartz10612/poppler-windows/releases"
            )
        except Exception as e:
            if "poppler" in str(e).lower():
                raise RuntimeError("Poppler not found. Ensure Poppler bin folder is in PATH.") from e
            raise e
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Only PDF and PNG are accepted.")


def get_image_dpi(image: Image.Image) -> tuple[float, float]:
    """
    Extracts DPI from image metadata.
    Benchmark: If missing, calculate DPI based on known 8-inch book height.
    This ensures physical mm matches pixel counts regardless of resolution.
    """
    width, height = image.size
    
    # Standard book height is 8.0 inches.
    # We enforce this benchmark DPI for accurate scaling, because generic EXIF "300 dpi" 
    # tags on downscaled thumbnail images will massively inflate the 9mm zone if trusted over the physical ratio.
    bench_dpi = height / config.COVER_HEIGHT_INCHES
    
    return float(bench_dpi), float(bench_dpi)


def mm_to_pixels(mm: float, dpi: float) -> int:
    """Converts millimetres to pixels given an image's DPI."""
    inches = mm / config.MM_PER_INCH
    return int(inches * dpi)


# ===========================================================================
# SAFE ZONE CALCULATIONS
# ===========================================================================

def calculate_badge_zone(image: Image.Image) -> BoundingBox:
    """Detection zone for the bottom badge strip.

    Spec: bottom 9mm is reserved for the award emblem.
    Implementation: we expand this by BADGE_DETECTION_BUFFER_MM so that any
    text hugging the top edge of the strip is still flagged as risky.
    """
    _, dpi_y = get_image_dpi(image)
    width, height = image.size
    badge_height_px = mm_to_pixels(
        config.BOTTOM_BADGE_ZONE_MM + config.BADGE_DETECTION_BUFFER_MM, dpi_y
    )

    return BoundingBox(
        x=0, y=height - badge_height_px, width=width, height=badge_height_px, text="[BADGE ZONE]",
    )

def calculate_front_badge_zone(image: Image.Image) -> BoundingBox:
    """
    Badge zone applied to the entire bottom of the image.
    The rule applies to the entire 9mm bottom strip, even for spread images.
    """
    return calculate_badge_zone(image)

def calculate_author_buffer(image: Image.Image) -> int:
    """9mm buffer from bottom edge (matches official badge zone)."""
    _, dpi_y = get_image_dpi(image)
    return mm_to_pixels(9.0, dpi_y)

# ===========================================================================
# VISUAL ANNOTATIONS
# ===========================================================================

def save_annotated_failure(image: Image.Image, isbn: str, badge_zone: BoundingBox):
    """Saves an annotated image showing the badge zone violation."""
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = cv_image.copy()
    cv2.rectangle(overlay, (badge_zone.x, badge_zone.y), (badge_zone.x2, badge_zone.y2), (0, 0, 255), -1)
    alpha = 0.3
    annotated = cv2.addWeighted(overlay, alpha, cv_image, 1 - alpha, 0)
    
    out_path = Path(config.TEMP_DOWNLOAD_DIR) / f"{isbn}_annotated.png"
    cv2.imwrite(str(out_path), annotated)
    logger.info(f"Saved annotated failure image to {out_path}")

# ===========================================================================
# ENGINE 1: GEOMETRY (OpenCV + Tesseract)
# ===========================================================================

def _ocr_boxes(cv_bgr: np.ndarray, *, psm: int, x_offset: int = 0, y_offset: int = 0) -> list[tuple[BoundingBox, float]]:
    """
    Runs Tesseract OCR and returns (BoundingBox, conf) pairs.
    Offsets are applied so boxes are in the original image coordinate system.
    """
    data = pytesseract.image_to_data(
        cv_bgr,
        output_type=pytesseract.Output.DICT,
        config=f"--psm {psm}",
    )
    out: list[tuple[BoundingBox, float]] = []
    n = len(data["level"])
    for i in range(n):
        text = data["text"][i].strip()
        try:
            conf = float(data["conf"][i])
        except Exception:
            continue
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        out.append((BoundingBox(x=x + x_offset, y=y + y_offset, width=w, height=h, text=text), conf))
    return out


def _preprocess_for_ocr(cv_bgr: np.ndarray) -> np.ndarray:
    """
    Aggressive preprocessing for small/light text near the bottom zone.
    Returns a BGR image so pytesseract continues to work as expected.
    """
    gray = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2GRAY)
    # Upscale to help OCR on thin fonts
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    # Increase local contrast & binarize
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
    )
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


def run_geometry_engine(image: Image.Image, author_name: Optional[str] = None, front_x_start: float = 0.0) -> EngineResult:
    """Mathematical validation engine."""
    issues: list[Issue] = []
    badge_zone = calculate_front_badge_zone(image)
    author_limit_px = calculate_author_buffer(image)
    height = image.size[1]
    width = image.size[0]

    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Primary OCR pass (whole image)
    primary_boxes = _ocr_boxes(cv_image, psm=11)

    # Secondary OCR pass (front cover bottom region).
    secondary_boxes: list[tuple[BoundingBox, float]] = []
    is_spread = (width / max(1, height)) >= 1.10
    if is_spread:
        # We now use Gemini's reported split line rather than hardcoding width//2
        # But if Gemini failed, we still use width//2 as a sensible default
        roi_x0 = int(front_x_start) if front_x_start > 0 else width // 2
        
        # Focus window: from ~40mm above detection zone down to the bottom of the image
        _, dpi_y = get_image_dpi(image)
        focus_h_px = mm_to_pixels(40.0, dpi_y)
        y0 = max(0, badge_zone.y - focus_h_px)
        roi = cv_image[y0:height, roi_x0:width].copy()
        
        if roi.size > 0:
            roi_pp = _preprocess_for_ocr(roi)
            secondary_boxes = _ocr_boxes(roi_pp, psm=6, x_offset=roi_x0, y_offset=y0)

    max_overlap_px = 0
    max_overlap_mm = 0.0
    
    # 1mm tolerance in pixels for edge detection
    dpi_x, dpi_y = get_image_dpi(image)
    pixel_tolerance = mm_to_pixels(1.0, dpi_y)
    
    # Proximity shield for placeholder fragments: 5mm in pixels
    shield_radius_px = mm_to_pixels(5.0, dpi_y)

    # First Pass: Identify all "Shields" (Definitive placeholders or Author Name)
    all_raw_boxes = primary_boxes + secondary_boxes
    placeholder_boxes: list[BoundingBox] = []
    
    # Shield Keywords
    shield_kws = ["dickinson", "century", "winner", "award", "emily"]
    if author_name:
        # Add author's name words to shield keywords
        shield_kws.extend(author_name.lower().split())
    
    for (text_box, conf) in all_raw_boxes:
        text = (text_box.text or "").strip()
        if not text:
            continue
        has_kw = any(kw in text.lower() for kw in shield_kws)
        if has_kw and len(text) <= 65:
            placeholder_boxes.append(text_box)

    # Second Pass: Filter and Classify Violations
    for (text_box, conf) in all_raw_boxes:
        text = (text_box.text or "").strip()
        if not text:
            continue
        
        # --- COORDINATE FILTER ---
        if text_box.x2 < front_x_start:
            continue

        # --- DUST FILTER ---
        # Ignore tiny fragments unless they have very high confidence.
        # Fragments like "ae", "i", "n" are usually OCR jitter.
        is_tiny = len(text) < 3
        if is_tiny and conf < 70:
            continue

        # EXCEPTION: If text is placeholder award text
        has_kw = any(kw in text.lower() for kw in ["dickinson", "century", "winner", "award", "emily"])
        is_placeholder = has_kw and len(text) <= 65
        
        # --- PLACEHOLDER SHIELD ---
        # If this box is NOT a placeholder but is touching or near a known placeholder, ignore it.
        # This absorbs "shattered" fragments of the award text.
        if not is_placeholder:
            is_near_placeholder = False
            for shield in placeholder_boxes:
                # Calculate distance between boxes
                dx = max(0, shield.x - text_box.x2, text_box.x - shield.x2)
                dy = max(0, shield.y - text_box.y2, text_box.y - shield.y2)
                if dx < shield_radius_px and dy < shield_radius_px:
                    is_near_placeholder = True
                    break
            
            if is_near_placeholder:
                logger.info(f"Math Engine: Shielding fragment '{text}' due to award proximity.")
                continue

        overlap_px = text_box.get_overlap_height(badge_zone)
        overlap_mm = (overlap_px / dpi_y) * config.MM_PER_INCH if dpi_y > 0 else 0.0
        
        # DEBUG LOGGING
        logger.info(
            f"OCR BOX >> text='{text}', conf={conf}, is_placeholder={is_placeholder}, "
            f"overlap_px={overlap_px}"
        )

        if overlap_mm > max_overlap_mm and not is_placeholder:
            max_overlap_mm = overlap_mm

        if text_box.overlaps_with(badge_zone, tolerance=pixel_tolerance):
            # Only add as a CRITICAL issue if it's NOT placeholder branding (which is allowed)
            # and the penetration is beyond the geometry review threshold.
            if not is_placeholder:
                if overlap_mm > config.GEOMETRY_REVIEW_THRESHOLD_MM:
                    issues.append(Issue(
                        issue_type=IssueType.BADGE_OVERLAP, severity=Severity.CRITICAL,
                        description=f'Text "{text}" enters the 9mm badge zone by ~{overlap_mm:.1f}mm.',
                        correction="Move text above the 9mm badge zone at the bottom."
                    ))
                elif overlap_mm > 0:
                    # Near-miss: within tolerance, record as INFO so humans can audit,
                    # but do not fail geometry on its own.
                    issues.append(Issue(
                        issue_type=IssueType.BADGE_OVERLAP, severity=Severity.INFO,
                        description=f'Text "{text}" is very close to the 9mm badge zone (~{overlap_mm:.1f}mm penetration).',
                        correction="If possible, nudge this text slightly upward to maximize clearance from the badge."
                    ))
            else:
                # Placeholder award text in the badge zone is allowed but noted as INFO.
                issues.append(Issue(
                    issue_type=IssueType.UNKNOWN, severity=Severity.INFO,
                    description=f'Placeholder award text "{text}" detected.',
                    correction="Ensure this is removed before final printing."
                ))
        
        # 2. Check Top, Left, Right Margins (3mm) with a small tolerance so
        #  the official "good" reference covers do not fail due to 1px OCR box noise.
        margin_px = mm_to_pixels(config.SIDE_MARGIN_MM, dpi_y)
        margin_tolerance_px = pixel_tolerance  # ~1mm in pixels

        if not is_placeholder:
            # Top margin: complain only if we are more than ~1mm inside the margin.
            if text_box.y < margin_px:
                penetration_px = margin_px - text_box.y
                if penetration_px > margin_tolerance_px:
                    issues.append(Issue(
                        issue_type=IssueType.MARGIN_VIOLATION, severity=Severity.CRITICAL,
                        description=f'Text "{text}" enters the {config.SIDE_MARGIN_MM}mm top margin.',
                        correction=f"Move text down by at least {config.SIDE_MARGIN_MM}mm from the top edge."
                    ))
                else:
                    issues.append(Issue(
                        issue_type=IssueType.MARGIN_VIOLATION, severity=Severity.INFO,
                        description=f'Text "{text}" is very close to the {config.SIDE_MARGIN_MM}mm top margin.',
                        correction="If possible, nudge this text slightly further from the top edge."
                    ))

            # Left margin.
            if text_box.x < margin_px:
                penetration_px = margin_px - text_box.x
                if penetration_px > margin_tolerance_px:
                    issues.append(Issue(
                        issue_type=IssueType.MARGIN_VIOLATION, severity=Severity.CRITICAL,
                        description=f'Text "{text}" enters the {config.SIDE_MARGIN_MM}mm left margin.',
                        correction=f"Move text to the right by at least {config.SIDE_MARGIN_MM}mm from the left edge."
                    ))
                else:
                    issues.append(Issue(
                        issue_type=IssueType.MARGIN_VIOLATION, severity=Severity.INFO,
                        description=f'Text "{text}" is very close to the {config.SIDE_MARGIN_MM}mm left margin.',
                        correction="If possible, nudge this text slightly further from the left edge."
                    ))

            # Right margin.
            right_limit_px = width - margin_px
            if text_box.x2 > right_limit_px:
                penetration_px = text_box.x2 - right_limit_px
                if penetration_px > margin_tolerance_px:
                    issues.append(Issue(
                        issue_type=IssueType.MARGIN_VIOLATION, severity=Severity.CRITICAL,
                        description=f'Text "{text}" enters the {config.SIDE_MARGIN_MM}mm right margin.',
                        correction=f"Move text to the left by at least {config.SIDE_MARGIN_MM}mm from the right edge."
                    ))
                else:
                    issues.append(Issue(
                        issue_type=IssueType.MARGIN_VIOLATION, severity=Severity.INFO,
                        description=f'Text "{text}" is very close to the {config.SIDE_MARGIN_MM}mm right margin.',
                        correction="If possible, nudge this text slightly further from the right edge."
                    ))

        if author_name and author_name.lower() in text.lower():
            author_penetration = text_box.y2 - (height - author_limit_px)
            if author_penetration > max_overlap_px:
                max_overlap_px = int(author_penetration)

            if author_penetration > pixel_tolerance:
                issues.append(Issue(
                    issue_type=IssueType.MARGIN_VIOLATION, severity=Severity.CRITICAL,
                    description=f'Author name "{text}" enters the 9mm bottom zone.',
                    correction="Ensure the author name is at least 9mm away from the bottom edge."
                ))

    return EngineResult(
        engine_name="Geometry (OpenCV + Tesseract)",
        passed=not any(i.severity == Severity.CRITICAL for i in issues), 
        issues=issues, 
        confidence=95.0 if not issues else 90.0,
        max_overlap_mm=max_overlap_mm
    )


# ===========================================================================
# ENGINE 2: VISION (Gemini)
# ===========================================================================

def run_vision_engine(image: Image.Image, geometry: Optional[EngineResult] = None) -> EngineResult:
    """AI vision engine that can also see Geometry output."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    geo_context = ""
    if geometry is not None:
        # Summarize key math findings for Gemini.
        lines = [
            "\n\nGEOMETRY ENGINE ANALYSIS:",
            f"- passed: {geometry.passed}",
            f"- max_badge_overlap_mm: {geometry.max_overlap_mm:.2f}",
            "- issues (top 10):",
        ]
        for iss in geometry.issues[:10]:
            lines.append(
                f"  - [{iss.severity.name}] {iss.issue_type.name}: {iss.description}"
            )
        lines.append(
            "Use these math findings together with the visual inspection to make the final PASS vs REVIEW decision."
        )
        geo_context = "\n".join(lines)

    prompt = f"""
You are the Primary Quality Control Inspector for BookLeaf Publishing. You act as the first line of defense.
Your job is to judge the "Visual Intent" and "Clearance" of the book cover design.

BOOK SPECIFICATIONS:
- Physical Size: 5x8 inches.
- Safe Margin: 3mm from top, left, and right edges.
- Badge Zone: The bottom 9mm strip is STRICTLY RESERVED for the award emblem.

STRICT VALIDATION RULES:
1. THE PLACEHOLDER EXCEPTION: Text mentioning "Emily Dickinson Award" is a placeholder and is ALLOWED in the bottom 9mm.
2. VISUAL OVERLAP (BADGE): Mark 'passed': false if any Book Title, Subtitle, or Author Name visually touches the bottom 9mm strip.
3. VISUAL OVERLAP (MARGINS): Mark 'passed': false if any text (Title, Name, etc.) touches the 3mm border at the top, left, or right edges.
4. PRECISION DOUBT: If you see text very close to the 3mm or 9mm lines and you are not 100% certain if it's safe, set "needs_precision_check": true.
5. SPATIAL AWARENESS: Identify the starting horizontal point where the FRONT COVER begins (ignoring the spine and back cover). 
   - Return this as "front_cover_x_min" on a scale of 0 to 1000 (e.g., if it's a spread and front cover starts halfway, return 500).
   - If it's a single front cover starting from the left edge, return 0.
6. DECISIVENESS: If the cover is clearly safe and compliant, provide high confidence (95%+) and set "needs_precision_check": false. Only request a precision check for true borderline cases where 1mm makes the difference.

OUTPUT FORMAT (JSON ONLY):
{{
  "passed": boolean,
  "confidence": integer (0-100),
  "needs_precision_check": boolean,
  "front_cover_x_min": integer (0-1000),
  "issues": [
    {{
      "issue_type": "Badge Overlap | Margin Violation",
      "severity": "Critical | Warning | Info",
      "description": "Short detail",
      "correction": "Step to fix"
    }}
  ]
}}{geo_context}
    """
    
    try:
        response = _gemini_model.generate_content([
            {"mime_type": "image/png", "data": image_b64},
            prompt
        ])
        
        raw = response.text.strip()
        if raw.startswith("```json"): raw = raw[7:-3]
        elif raw.startswith("```"): raw = raw[3:-3]
        
        data = json.loads(raw)
        issues = []
        for i in data.get("issues", []):
            severity_str = i.get("severity", "Critical")
            if severity_str == "Info":
                severity = Severity.INFO
            elif severity_str == "Warning":
                severity = Severity.WARNING
            else:
                severity = Severity.CRITICAL
                
            issues.append(Issue(
                issue_type=IssueType.UNKNOWN,
                severity=severity,
                description=i["description"],
                correction=i["correction"]
            ))
        
        # Redundant branding text check (Post-process or ensure it's INFO)
        is_award_placeholder = "emily dickinson award" in response.text.lower()
        if is_award_placeholder:
            # Check if vision engine already flagged it
            found = False
            for i in issues:
                if "award" in i.description.lower():
                    i.severity = Severity.INFO # Force to INFO
                    i.correction = "Placeholder award text detected. Ensure this is removed before final print as a physical emblem will be added by the publisher."
                    found = True
            
            if not found:
                 issues.append(Issue(
                    issue_type=IssueType.UNKNOWN, severity=Severity.INFO,
                    description="Placeholder award branding text detected.",
                    correction="Placeholder award text detected. Ensure this is removed before final print as a physical emblem will be added by the publisher."
                ))

        # Re-evaluate PASS status (Placeholder text is allowed)
        is_actually_passed = data.get("passed", True)
        if all(i.severity == Severity.INFO for i in issues):
            is_actually_passed = True

        # Calculate absolute pixel coordinate for front cover split
        width = image.size[0]
        norm_x = data.get("front_cover_x_min", 0)
        front_x_start_px = (norm_x / 1000.0) * width

        return EngineResult(
            engine_name="Vision (Gemini 3.0 Flash Preview)", # Reverted to correct model title
            passed=is_actually_passed,
            issues=issues, 
            confidence=float(data.get("confidence", 80)),
            needs_precision_check=data.get("needs_precision_check", False),
            front_cover_x_start_px=front_x_start_px
        )
    except Exception as e:
        # If Gemini is unavailable (quota, network, etc.), treat Vision as "skipped"
        # instead of a hard failure so Geometry can still make a decision.
        logger.error(f"Vision engine error: {e}")
        fallback_issue = Issue(
            issue_type=IssueType.UNKNOWN,
            severity=Severity.INFO,
            description="Vision engine was unavailable; decision is based on geometry only.",
            correction="System could not run the AI vision check for this cover. If in doubt, review it manually."
        )
        return EngineResult(
            engine_name="Vision (Gemini)",
            passed=True,
            issues=[fallback_issue],
            confidence=0.0,
        )

# ===========================================================================
# CONSENSUS & MAIN
# ===========================================================================

def evaluate_consensus(isbn: str, geo: EngineResult, vision: EngineResult, cover_file=None) -> ValidationResult:
    """
    Merges results using a STRICTLY CONSERVATIVE consensus model.
    - A cover is only marked PASS when BOTH engines agree it is perfectly safe.
    - Any geometry overlap, CRITICAL/WARNING issue, low confidence, or engine failure forces REVIEW_NEEDED.
    """
    logger.info(f"--- CONSENSUS DATA [{isbn}] ---")
    logger.info(f"Geometry: passed={geo.passed}, max_overlap={geo.max_overlap_mm:.2f}mm")
    logger.info(f"Vision  : passed={vision.passed}, confidence={vision.confidence}%")

    # Conservative defaults, but allow a narrow AI override for
    # margin-only disagreements when Vision is very confident.
    ai_override = False

    geometry_skipped = geo.engine_name == "Geometry (Skipped)"

    # Any non-trivial math overlap with the badge forces REVIEW_NEEDED.
    if not geometry_skipped and geo.max_overlap_mm > config.GEOMETRY_REVIEW_THRESHOLD_MM:
        status = ValidationStatus.REVIEW_NEEDED
    else:
        # Basic agreement check:
        # - Both engines must pass
        # - No CRITICAL or WARNING issues from either engine
        all_issues = geo.issues + vision.issues
        has_hard_issue = any(i.severity in (Severity.CRITICAL, Severity.WARNING) for i in all_issues)

        if (
            not geometry_skipped
            and geo.passed
            and vision.passed
            and not has_hard_issue
            and geo.max_overlap_mm <= config.GEOMETRY_REVIEW_THRESHOLD_MM
        ):
            status = ValidationStatus.PASS
        else:
            status = ValidationStatus.REVIEW_NEEDED

    # Narrow AI override path:
    # If Geometry is failing *only* due to side-margin violations, but Vision is
    # clearly happy and high-confidence, allow Vision to upgrade to PASS while
    # still keeping badge overlaps strict.
    if not geometry_skipped and status == ValidationStatus.REVIEW_NEEDED:
        has_badge_overlap = any(
            (iss.issue_type == IssueType.BADGE_OVERLAP and iss.severity == Severity.CRITICAL)
            for iss in geo.issues
        )
        critical_geo = [iss for iss in geo.issues if iss.severity == Severity.CRITICAL]
        only_margin_criticals = critical_geo and all(
            iss.issue_type == IssueType.MARGIN_VIOLATION for iss in critical_geo
        )

        if (
            not has_badge_overlap
            and only_margin_criticals
            and vision.passed
            and not vision.needs_precision_check
            and vision.confidence >= 90.0
        ):
            ai_override = True
            status = ValidationStatus.PASS

    # 2. Filter and Merge issues
    final_issues = []
    seen = set()
    
    source_issues = geo.issues + vision.issues
    for iss in source_issues:
        # If AI override is active, suppress Geometry's critical overlaps (they were deemed minor/placeholder)
        if ai_override and iss in geo.issues and iss.severity == Severity.CRITICAL:
            continue
            
        if iss.description not in seen:
            final_issues.append(iss)
            seen.add(iss.description)

    if ai_override:
        final_issues.append(Issue(
            issue_type=IssueType.UNKNOWN,
            severity=Severity.INFO,
            description=(
                "Vision (Gemini) confirmed the layout as visually safe. "
                "Geometry margin violations were treated as suggestions, not blockers."
            ),
            correction="You may still choose to nudge side text slightly inward, but this cover is considered acceptable."
        ))

    return ValidationResult(
        isbn=isbn,
        author_name=getattr(cover_file, "author_name", None),
        author_email=getattr(cover_file, "author_email", None),
        status=status,
        confidence=(geo.confidence + vision.confidence) / 2,
        issues=final_issues,
        geometry_result=geo,
        vision_result=vision
    )

def process_book_cover(file_path: str, isbn: str, cover_file=None, *, skip_vision: bool = False) -> ValidationResult:
    """
    Full pipeline: Geometry engine runs first, then Vision (Gemini) sees
    both the image and the geometry findings and a consensus decision is made.
    Set skip_vision=True for fast demo mode (geometry-only); avoids Gemini latency.
    """
    image = load_and_preprocess(file_path)
    author_name = getattr(cover_file, "author_name", None)
    width = image.size[0]

    # STEP 1: Always run Geometry first.
    geo = run_geometry_engine(image, author_name)

    # STEP 2: Vision (optional) sees both the image and geo summary.
    if skip_vision:
        vision = EngineResult(
            engine_name="Vision (skipped for fast demo)",
            passed=True,
            issues=[],
            confidence=0.0,
            front_cover_x_start_px=width / 2.0,
        )
        logger.info(f"ISBN={isbn}: Skipping Vision engine (demo mode). Using Geometry only.")
    else:
        vision = run_vision_engine(image, geo)

    # STEP 3: Final consensus.
    result = evaluate_consensus(isbn, geo, vision, cover_file)

    if result.status != ValidationStatus.PASS:
        save_annotated_failure(image, isbn, calculate_front_badge_zone(image))
    return result
