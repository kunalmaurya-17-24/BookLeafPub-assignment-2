import os
from pathlib import Path

import gradio as gr

import config
from detection import process_book_cover
from schemas import CoverFile, FileType


def _infer_file_type(path: str) -> FileType:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return FileType.PDF
    return FileType.PNG


def validate_cover(file, isbn: str, author_name: str, author_email: str):
    """
    Gradio callback: validates a single uploaded cover.
    """
    if file is None:
        return "No file uploaded.", "", "", None

    temp_path = file.name
    isbn = (isbn or "").strip() or "UNKNOWN"
    author_name = (author_name or "").strip() or None
    author_email = (author_email or "").strip() or None

    cover = CoverFile(
        isbn=isbn,
        file_type=_infer_file_type(temp_path),
        filename=Path(temp_path).name,
        file_path=temp_path,
        author_name=author_name,
        author_email=author_email,
    )

    result = process_book_cover(temp_path, isbn, cover)

    status_text = f"Status: {result.status.value}  |  Confidence: {result.confidence:.1f}%"
    issues_text = result.formatted_issues
    corrections_text = result.formatted_corrections

    # If the cover failed, an annotated image is saved in TEMP_DOWNLOAD_DIR
    annotated_path = Path(config.TEMP_DOWNLOAD_DIR) / f"{isbn}_annotated.png"
    annotated_image = str(annotated_path) if annotated_path.exists() else None

    return status_text, issues_text, corrections_text, annotated_image


with gr.Blocks() as app:
    gr.Markdown(
        "## BookLeaf Cover Validation — Demo\n"
        "Upload a cover (PNG/PDF), enter ISBN and author info, then run validation.\n\n"
        "*This demo runs the **full pipeline**: Geometry (math rules) + Gemini Vision. "
        "Geometry enforces the 3mm/9mm rules; Gemini double-checks the actual design before PASS/REVIEW.*"
    )

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Cover file (PNG or PDF)")
            isbn_input = gr.Textbox(label="ISBN", placeholder="9789373...")
            author_name_input = gr.Textbox(label="Author name (optional)")
            author_email_input = gr.Textbox(label="Author email (optional)")
            run_button = gr.Button("Validate Cover", variant="primary")

        with gr.Column():
            status_output = gr.Textbox(label="Validation status", interactive=False)
            issues_output = gr.Textbox(label="Detected issues", lines=8, interactive=False)
            corrections_output = gr.Textbox(label="Correction instructions", lines=8, interactive=False)
            annotated_output = gr.Image(label="Annotated badge zone (for non-PASS results)", visible=True)

    run_button.click(
        fn=validate_cover,
        inputs=[file_input, isbn_input, author_name_input, author_email_input],
        outputs=[status_output, issues_output, corrections_output, annotated_output],
    )


if __name__ == "__main__":
    app.launch()

