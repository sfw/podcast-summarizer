import os
import math
import logging
import shutil
import tempfile
import requests
import gradio as gr

from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI

##############################
# Load environment & logging
##############################
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)
MAX_CHUNK_SIZE = 24 * 1024 * 1024  # ~24 MB

# The user can select any combination of these in the UI:
SELECTABLE_OPTIONS = ["Summary", "Keywords", "Titles"]

##############################
# Audio Splitting & Whisper
##############################
def split_audio_to_chunks(audio_file_path: str, max_chunk_size_bytes: int) -> list[str]:
    logger.info(f"Splitting audio: {audio_file_path}")
    audio = AudioSegment.from_file(audio_file_path)

    total_duration_ms = len(audio)
    total_bytes = len(audio.raw_data)
    if total_bytes == 0:  # Edge case: empty or unreadable file
        return []

    chunk_duration_ms = math.floor((max_chunk_size_bytes / total_bytes) * total_duration_ms)
    if chunk_duration_ms < 1000:
        chunk_duration_ms = 60_000  # fallback to 60 seconds

    chunk_paths = []
    start_ms = 0
    while start_ms < total_duration_ms:
        end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
        chunk_audio = audio[start_ms:end_ms]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            chunk_audio.export(tmp.name, format="wav")
            chunk_paths.append(tmp.name)

        start_ms = end_ms

    logger.info(f"Split into {len(chunk_paths)} chunk(s).")
    return chunk_paths

def whisper_translation_request(chunk_path: str) -> str:
    if not OPENAI_API_KEY:
        return "[Error: OPENAI_API_KEY not set.]"

    url = "https://api.openai.com/v1/audio/translations"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"model": "whisper-1"}

    with open(chunk_path, "rb") as f:
        files = {"file": (chunk_path, f, "application/octet-stream")}
        try:
            resp = requests.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
            result = resp.json()
            return result.get("text", "")
        except requests.RequestException as e:
            logger.error(f"Transcription request failed: {e}")
            return f"[Error transcribing chunk: {e}]"

##############################
# Main Processing Function
##############################
def process_audio_files(
    audio_file_paths,
    selected_options,
    summary_prompt_text,
    keywords_prompt_text,
    titles_prompt_text
):
    """
    1. For each uploaded file:
       - Split into <25MB chunks
       - Whisper-translate them all
       - Conditionally produce Summary, Keywords, Titles via Chat calls
         (based on 'selected_options'), using the user-provided prompt text
         plus the transcript at the end.
       - Save each output in a directory named after the file's base name
    2. Build a dynamic HTML snippet with a "tab" for each file
       (one tab containing that file's transcript, plus whichever of the
       summary, keywords, or titles the user selected).
    3. Return:
       - status updates (via yield)
       - final HTML to display
    """
    yield "Starting transcription...", ""
    if not audio_file_paths or len(audio_file_paths) == 0:
        return "<p>No files provided.</p>"
    if not OPENAI_API_KEY:
        return "<p>OPENAI_API_KEY not set!</p>"

    css_and_script = """
<style>
.tab-container {
  display: flex;
  flex-direction: column;
}
.tab-buttons {
  display: flex;
  flex-wrap: nowrap;
  overflow-x: auto;
  max-width: 100%;
}
.tab-buttons label {
  padding: 8px 12px;
  margin-right: 4px;
  cursor: pointer;
  border-radius: 4px 4px 0 0;
  white-space: nowrap;
  text-overflow: ellipsis;
  text-wrap: auto;
  overflow: hidden;
  max-width: 150px;
  display: inline-block;
  box-sizing: border-box;
  word-wrap: break-word;
}
.tab-content pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  max-width: 100%;
  overflow-x: auto;
}
.tab-content {
  border: 1px solid #ddd;
  padding: 10px;
  display: none;
  position: relative;
  border-radius: 0 4px 4px 4px;
}
input[type="radio"] {
  display: none !important;
}
input[type="radio"]:checked + label {
  background: #ccc;
  color: #000;
}
input[type="radio"]:checked + label + .tab-content {
  display: block;
}
</style>
"""

    # 1) Build a parent folder name using the number of files and current date/time
    parent_name = f"Summarized-{len(audio_file_paths)}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    parent_folder = Path(parent_name)
    parent_folder.mkdir(exist_ok=True)

    html_tabs = ['<div class="tab-container">', '<div class="tab-buttons">']

    for i, local_path in enumerate(audio_file_paths):
        base_name = Path(local_path).stem
        tab_id = f"tab_{i}"

        if not os.path.exists(local_path):
            content_html = f"<p>File not found: {local_path}</p>"
        else:
            folder_path = parent_folder / base_name
            folder_path.mkdir(exist_ok=True)

            # (1) Split
            chunk_paths = split_audio_to_chunks(local_path, MAX_CHUNK_SIZE)
            # (2) Whisper
            transcript_parts = []
            for idx, cp in enumerate(chunk_paths, start=1):
                logger.info(f"Transcribing {base_name}, chunk {idx}/{len(chunk_paths)}")
                yield f"Transcribing {base_name}, chunk {idx}/{len(chunk_paths)}", ""
                chunk_text = whisper_translation_request(cp)
                transcript_parts.append(chunk_text)
                try:
                    os.remove(cp)
                except OSError:
                    pass

            final_transcript = "\n".join(transcript_parts)
            (folder_path / "transcript.txt").write_text(final_transcript, encoding="utf-8")

            # (3) Summaries, Keywords, Titles (conditionally)
            model_engine = "o1-mini"

            # SUMMARY
            try:
                if "Summary" in selected_options:
                    # Combine user-provided prompt + transcript
                    prompt_with_transcript = (
                        summary_prompt_text.strip() +
                        f"\n\nTRANSCRIPT:\n{final_transcript}\n"
                    )
                    summary_resp = client.chat.completions.create(
                        model=model_engine,
                        messages=[{"role": "user", "content": prompt_with_transcript}],
                        temperature=1
                    )
                    summarized_description = summary_resp.choices[0].message.content.strip()
                    (folder_path / "summary.txt").write_text(summarized_description, encoding="utf-8")
                else:
                    summarized_description = "[Summary was not generated]"
            except Exception as e:
                summarized_description = f"[Error generating summary: {e}]"
                logger.error(summarized_description)
                yield summarized_description, ""

            # KEYWORDS
            try:
                if "Keywords" in selected_options:
                    prompt_with_transcript = (
                        keywords_prompt_text.strip() +
                        f"\n\nTRANSCRIPT:\n{final_transcript}\n"
                    )
                    keywords_resp = client.chat.completions.create(
                        model=model_engine,
                        messages=[{"role": "user", "content": prompt_with_transcript}],
                        temperature=1
                    )
                    seo_keywords = keywords_resp.choices[0].message.content.strip()
                    (folder_path / "keywords.txt").write_text(seo_keywords, encoding="utf-8")
                else:
                    seo_keywords = "[Keywords were not generated]"
            except Exception as e:
                seo_keywords = f"[Error generating keywords: {e}]"
                logger.error(seo_keywords)
                yield seo_keywords, ""

            # TITLES
            try:
                if "Titles" in selected_options:
                    prompt_with_transcript = (
                        titles_prompt_text.strip() +
                        f"\n\nTRANSCRIPT:\n{final_transcript}\n"
                    )
                    titles_resp = client.chat.completions.create(
                        model=model_engine,
                        messages=[{"role": "user", "content": prompt_with_transcript}],
                        temperature=1
                    )
                    video_titles = titles_resp.choices[0].message.content.strip()
                    (folder_path / "titles.txt").write_text(video_titles, encoding="utf-8")
                else:
                    video_titles = "[Titles were not generated]"
            except Exception as e:
                video_titles = f"[Error generating titles: {e}]"
                logger.error(video_titles)
                yield video_titles, ""

            # (4) Build the HTML for the content of this tab
            content_html = f"""
<h3>Transcript</h3>
<pre>{final_transcript}</pre>
{"<h3>Summary</h3><pre>"+ summarized_description + "</pre>" if "Summary" in selected_options else ""}
{"<h3>SEO Keywords</h3><pre>"+ seo_keywords + "</pre>" if "Keywords" in selected_options else ""}
{"<h3>Video Thumbnail Titles</h3><pre>"+ video_titles + "</pre>" if "Titles" in selected_options else ""}
<p><em>Outputs also saved in: {folder_path}</em></p>
"""

        checked = "checked" if i == 0 else ""
        html_tabs.append(f'<input type="radio" id="{tab_id}" name="tabs" {checked}>')
        html_tabs.append(f'<label for="{tab_id}">{Path(local_path).name}</label>')
        html_tabs.append(f'<div class="tab-content">{content_html}</div>')

    html_tabs.append("</div>")  # close .tab-buttons
    html_tabs.append("</div>")  # close .tab-container

    # Zip the entire parent folder
    zip_path = shutil.make_archive(parent_folder.name, "zip", parent_folder)
    download_link = f'<a href="file://{os.path.abspath(zip_path)}" download>Download Zip</a>'

    full_html = css_and_script + "\n".join(html_tabs)
    yield f"Processing complete, rendering HTML - Download the files: {download_link}", full_html

##############################
# Gradio App
##############################
with gr.Blocks(css=".footer.light {display: none !important;}") as demo:
    gr.Markdown(
        "# Podcast Audio Summarizer\n"
        "Upload any number of files, each file gets its own tab and every batch is "
        "saved out in structured folders as .txt files. You can also select which outputs "
        "you want generated and **edit** the prompt text for each option before processing."
    )
    with gr.Row():
        audio_input = gr.File(
            label="Upload Audio Files",
            file_count="multiple",
            type="filepath"
        )
        selected_options = gr.CheckboxGroup(
            choices=SELECTABLE_OPTIONS,
            value=["Summary", "Keywords", "Titles"],
            label="Select Which Outputs"
        )

    # Create a row with the "Process" button on top, then 3 columns of text fields
    with gr.Row():
        submit_btn = gr.Button("Process")

    # Under that row, we have 3 columns for the prompts
    with gr.Row():
        with gr.Column():
            summary_prompt_text = gr.Textbox(
                label="Summary Prompt",
                value=(
                    "Provide me a summarized description of this transcript from a podcast episode "
                    "for a video description. Make the length reasonable but as long as you need to "
                    "be detailed. Focus on including keywords in the summary that would impact SEO, "
                    "but do not add a separate list of explicit keywords. Return plain text, no markdown, no bold text.\n\n"
                    "Also, be aware the proper spelling for the two hosts' names are "
                    "Jordan Bloemen and Scott Francis Winder."
                ),
                lines=8
            )
        with gr.Column():
            keywords_prompt_text = gr.Textbox(
                label="Keywords Prompt",
                value=(
                    "From the following transcript, provide a comma-separated list of top relevant keywords "
                    "that would improve SEO. Focus on main subjects and terms. Return plain text, no markdown, no bold text."
                ),
                lines=8
            )
        with gr.Column():
            titles_prompt_text = gr.Textbox(
                label="Titles Prompt",
                value=(
                    "Provide five short video thumbnail title recommendations (max 6 words each) "
                    "from this transcript. Return plain text, no markdown, no bold text."
                ),
                lines=8
            )

    status_box = gr.HTML(label="Status")
    html_output = gr.HTML()

    # We disable the button, call process_audio_files, then re-enable
    submit_btn.click(
        fn=lambda: gr.update(interactive=False),
        inputs=None,
        outputs=submit_btn,
        queue=False
    ).then(
        fn=process_audio_files,
        inputs=[
            audio_input,
            selected_options,
            summary_prompt_text,
            keywords_prompt_text,
            titles_prompt_text
        ],
        outputs=[status_box, html_output]
    ).then(
        fn=lambda: gr.update(interactive=True),
        inputs=None,
        outputs=submit_btn
    )

demo.launch(share=True)
