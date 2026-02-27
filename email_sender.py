"""
email_sender.py
===============
Sends personalized cover validation emails to authors via Gmail SMTP.

Why SMTP (not Gmail API or Service Account)?
  - Service accounts cannot send Gmail on personal @gmail.com accounts.
  - Gmail API requires OAuth which forces the evaluator to sign in.
  - SMTP + App Password is fully automated, requires ONLY an env variable,
    no Google Cloud setup, and works out of the box.

Setup (one-time, for the sender):
  1. Enable 2-Step Verification on the sending Gmail account.
  2. Go to: Google Account → Security → App Passwords.
  3. Generate an App Password for "Mail" → copy the 16-character code.
  4. Set EMAIL_SENDER and EMAIL_APP_PASSWORD in your .env file.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config
from schemas import ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)

GMAIL_SMTP_HOST = "smtp.gmail.com"
GMAIL_SMTP_PORT = 587


# ---------------------------------------------------------------------------
# Email body builder
# ---------------------------------------------------------------------------

def build_email_body(result: ValidationResult) -> tuple[str, str]:
    """
    Builds the email subject and HTML body for a ValidationResult.
    Returns (subject, html_body).
    """
    author = result.author_name or "Author"
    isbn = result.isbn
    status = result.status.value
    confidence = f"{result.confidence:.1f}%"

    if result.status == ValidationStatus.PASS:
        # ------------------------------------------------------------------
        # PASS email
        # ------------------------------------------------------------------
        subject = f"✅ Cover Approved — ISBN {isbn} | BookLeaf Publishing"
        html = f"""
<html><body style="font-family: Arial, sans-serif; color: #222; max-width: 600px;">
  <h2 style="color: #28a745;">✅ Your Book Cover Has Been Approved</h2>

  <p>Dear {author},</p>

  <p>Great news! Our automated cover validation system has reviewed your submission
  and <strong>all layout checks have passed</strong>.</p>

  <table style="border-collapse:collapse; width:100%; margin: 16px 0;">
    <tr><td style="padding:6px 12px; background:#f5f5f5; font-weight:bold;">ISBN</td>
        <td style="padding:6px 12px;">{isbn}</td></tr>
    <tr><td style="padding:6px 12px; background:#f5f5f5; font-weight:bold;">Status</td>
        <td style="padding:6px 12px; color:#28a745;"><strong>{status}</strong></td></tr>
    <tr><td style="padding:6px 12px; background:#f5f5f5; font-weight:bold;">Confidence</td>
        <td style="padding:6px 12px;">{confidence}</td></tr>
  </table>

  <p>Your cover has been queued for the next stage of the publishing workflow.
  You will receive a separate communication regarding printing timelines.</p>

  <p style="margin-top:24px;">Best regards,<br>
  <strong>BookLeaf Publishing — Automated Review System</strong><br>
  Support: <a href="mailto:{config.SUPPORT_EMAIL}">{config.SUPPORT_EMAIL}</a></p>
</body></html>
"""
    else:
        # ------------------------------------------------------------------
        # REVIEW NEEDED email
        # ------------------------------------------------------------------
        subject = f"⚠️ Cover Review Required — ISBN {isbn} | BookLeaf Publishing"

        # Build issues section
        issue_lines = []
        for issue in result.issues:
            icon = "❌" if issue.severity.value == "Critical" else "⚠️"
            issue_lines.append(
                f"<li style='margin-bottom:8px;'>"
                f"<strong>{icon} {issue.issue_type.value}</strong><br>"
                f"{issue.description}"
                f"</li>"
            )
        issues_html = "<ul>" + "".join(issue_lines) + "</ul>" if issue_lines else "<p>See corrections below.</p>"

        # Build corrections section
        correction_lines = []
        for i, issue in enumerate(result.issues, 1):
            correction_lines.append(
                f"<li style='margin-bottom:8px;'><strong>Step {i}:</strong> {issue.correction}</li>"
            )
        corrections_html = "<ol>" + "".join(correction_lines) + "</ol>" if correction_lines else ""

        html = f"""
<html><body style="font-family: Arial, sans-serif; color: #222; max-width: 600px;">
  <h2 style="color: #dc3545;">⚠️ Action Required: Cover Review Needed</h2>

  <p>Dear {author},</p>

  <p>Our automated cover validation system has reviewed your submission for
  <strong>ISBN {isbn}</strong> and has identified layout issues that require
  correction before your cover can be approved for printing.</p>

  <table style="border-collapse:collapse; width:100%; margin: 16px 0;">
    <tr><td style="padding:6px 12px; background:#f5f5f5; font-weight:bold;">ISBN</td>
        <td style="padding:6px 12px;">{isbn}</td></tr>
    <tr><td style="padding:6px 12px; background:#f5f5f5; font-weight:bold;">Status</td>
        <td style="padding:6px 12px; color:#dc3545;"><strong>{status}</strong></td></tr>
    <tr><td style="padding:6px 12px; background:#f5f5f5; font-weight:bold;">Confidence</td>
        <td style="padding:6px 12px;">{confidence}</td></tr>
  </table>

  <h3>Issues Detected</h3>
  {issues_html}

  <h3>How to Fix</h3>
  {corrections_html}

  <h3>Resubmission</h3>
  <p>Please apply the corrections above and <strong>re-upload your cover to the
  same Google Drive folder within 3 business days</strong>, using the same
  <code>ISBN_text.extension</code> filename format.</p>

  <p>Our system will automatically re-validate your updated submission.</p>

  <hr style="margin-top:32px;">
  <p style="font-size:12px; color:#666;">
    If you believe this is an error or need assistance, contact:<br>
    <a href="mailto:{config.SUPPORT_EMAIL}">{config.SUPPORT_EMAIL}</a><br>
    BookLeaf Publishing — Automated Cover Review System
  </p>
</body></html>
"""

    return subject, html


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

def send_email(to_email: str, subject: str, html_body: str) -> bool:
    """
    Sends an HTML email via Gmail SMTP.

    Returns True on success, False on failure (never raises — failures are logged).
    """
    if not config.EMAIL_SENDER or not config.EMAIL_APP_PASSWORD:
        logger.error("Email not configured. Set EMAIL_SENDER and EMAIL_APP_PASSWORD in .env")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"BookLeaf Publishing <{config.EMAIL_SENDER}>"
        msg["To"] = to_email
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(GMAIL_SMTP_HOST, GMAIL_SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(config.EMAIL_SENDER, config.EMAIL_APP_PASSWORD)
            server.sendmail(config.EMAIL_SENDER, to_email, msg.as_string())

        logger.info(f"[Email] Sent to {to_email} | Subject: {subject}")
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error(
            "[Email] Authentication failed. Check EMAIL_APP_PASSWORD in .env. "
            "Make sure you're using an App Password, not your Gmail login password."
        )
        return False
    except Exception as e:
        logger.error(f"[Email] Failed to send to {to_email}: {e}")
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def notify_author(result: ValidationResult) -> bool:
    """
    Builds and sends the appropriate email for a ValidationResult.

    Uses the author_email from the ValidationResult.
    Returns True if the email was sent successfully.
    """
    if not result.author_email:
        logger.warning(
            f"[Email] No author email found for ISBN={result.isbn}. Skipping notification."
        )
        return False

    subject, html_body = build_email_body(result)
    return send_email(result.author_email, subject, html_body)
