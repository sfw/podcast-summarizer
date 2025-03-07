import os
import math
import logging
import shutil
import tempfile
import requests
import assemblyai as aai
import gradio as gr

from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from pydub import AudioSegment
from openai import OpenAI
from google import genai
from prompt_refine_lib import add_prompt_refinement_modal

##############################
# Load environment & logging
##############################
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

aai.settings.api_key = ASSEMBLYAI_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

MAX_CHUNK_SIZE = 24 * 1024 * 1024  # ~24 MB

# The user can select any combination of these in the UI:
SELECTABLE_OPTIONS = ["Summary", "Keywords", "Titles", "Shorts"]
TRANSCRIPTION_OPTIONS = ["AssemblyAI - Speaker ID", "OpenAI Whisper"]


##############################
# Generic Call to OpenAI Library
##############################
def openai_prompt_call(
        model_data,
        prompt_text,
        final_transcript
):
    """
    """
    prompt_with_transcript = (
        prompt_text.strip() +
        f"\n\nTRANSCRIPT:\n{final_transcript}\n"
    )
    summary_resp = model_data["client"].chat.completions.create(
        model=model_data["model"],
        messages=[{"role": model_data["role"], "content": prompt_with_transcript}],
        temperature=model_data["temperature"]
    )
    return summary_resp.choices[0].message.content.strip()


##############################
# Generic Call to Gemini Library
##############################
def gemini_prompt_call(
        model_data,
        prompt_text,
        final_transcript
):
    """
    """
    prompt_with_transcript = (
        prompt_text.strip() +
        f"\n\nTRANSCRIPT:\n{final_transcript}\n"
    )
    summary_resp = model_data["client"].models.generate_content(model=model_data["model"], contents = prompt_with_transcript)
    return summary_resp.text



ENGINE_OPTIONS = {
    "OpenAI-o1mini":   {
                    "function": openai_prompt_call,
                    "client": openai_client,
                    "model": "o1-mini",
                    "temperature": 1,
                    "role": "user"
                },
    # "DeepSeek-V3": {
    #                 "function": openai_prompt_call,
    #                 "client": deepseek_client,
    #                 "model": "deepseek-chat",
    #                 "temperature": 1,
    #                 "role": "user"
    #             },
    # "DeepSeek-R1": {
    #                 "function": openai_prompt_call,
    #                 "client": deepseek_client,
    #                 "model": "reasoner",
    #                 "temperature": 1,
    #                 "role": "user"
    #             },
    "Gemini-2.0": {
                    "function": gemini_prompt_call,
                    "client": gemini_client,
                    "model": "gemini-2.0-flash-exp"
                },  
    "Gemini-1.5": {
                    "function": gemini_prompt_call,
                    "client": gemini_client,
                    "model": "gemini-1.5-flash"
                }          
}

##############################
# Audio Splitting & Transcription
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

# OpenAI Whisper if Preferred
def transcription_request_whisper(chunk_path: str) -> str:
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

def transcription_request_assembly(chunk_path: str) -> str:
    if not ASSEMBLYAI_API_KEY:
        return "[Error: ASSEMBLYAI_API_KEY not set.]"
    
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber(config=config)
    
    try:
        transcript = transcriber.transcribe(chunk_path)
        transcript.wait_for_completion()  # Wait until transcription is complete
        
        result = ""
        if hasattr(transcript, 'utterances') and transcript.utterances:
            for utterance in transcript.utterances:
                result += f"Speaker {utterance.speaker}: {utterance.text}\n\n"
        else:
            result = transcript.text
        
        return result
    except Exception as e:
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
    titles_prompt_text,
    shorts_prompt_text,
    transcription_engine,
    summary_engine_index,
    keywords_engine_index,
    titles_engine_index,
    shorts_engine_index
):
    """
    1. For each uploaded file:
       - Split into <25MB chunks
       - Translate them all
       - Conditionally produce Summary, Keywords, Titles, Shorts via Chat calls
         (based on 'selected_options'), using the user-provided prompt text
         plus the transcript at the end.
       - Save each output in a directory named after the file's base name
    2. Build a dynamic HTML snippet with a "tab" for each file
       (one tab containing that file's transcript, plus whichever of the
       summary, keywords, titles, or shorts the user selected).
    3. Return:
       - status updates (via yield)
       - final HTML to display
    """
    yield "Starting transcription...", "", None  # Initialize with None for the download link

    if not audio_file_paths or len(audio_file_paths) == 0:
        return "<p>No files provided.</p>", None, None
    if not OPENAI_API_KEY:
        return "<p>OPENAI_API_KEY not set!</p>", None, None

    css_and_script = """
    <style>
    /* Basic styling for the tab system */
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
      background: var(--block-background-fill);
    }
    .tab-content pre {
      white-space: pre-wrap;
      word-wrap: break-word;
      max-width: 100%;
      overflow-x: auto;
    }
    .tab-content {
      border: 1px solid var(--input-background-fill);
      padding: 10px;
      display: none;
      position: relative;
      border-radius: 0 4px 4px 4px;
    }
    input[type="radio"] {
      display: none !important;
    }
    input[type="radio"]:checked + label {
      background: var(--input-background-fill);
      color: #fff;
    }
    input[type="radio"]:checked + label + .tab-content {
      display: block;
      background: var(--input-background-fill);
      color: #fff;
    }
    </style>
    """

    # 1) Build a parent folder name using the number of files and current date/time
    parent_name = f"Summarized-{len(audio_file_paths)}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    parent_folder = Path(parent_name).absolute()
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
            # (2) Transcription
            transcript_parts = []
            for idx, cp in enumerate(chunk_paths, start=1):
                logger.info(f"Transcribing {base_name}, chunk {idx}/{len(chunk_paths)}")
                yield f"Transcribing {base_name}, chunk {idx}/{len(chunk_paths)}", "", None
                chunk_text=None;
                if transcription_engine == 0:
                    chunk_text = transcription_request_assembly(cp)
                else:
                    chunk_text = transcription_request_whisper(cp)
                transcript_parts.append(chunk_text or "")
                try:
                    os.remove(cp)
                except OSError:
                    pass

            final_transcript = "\n".join(transcript_parts)
            (folder_path / "transcript.txt").write_text(final_transcript, encoding="utf-8")

            
            
            # Grab the relevant engine for each query
            summary_engine = ENGINE_OPTIONS[summary_engine_index]
            keywords_engine = ENGINE_OPTIONS[keywords_engine_index]
            titles_engine = ENGINE_OPTIONS[titles_engine_index]
            shorts_engine = ENGINE_OPTIONS[shorts_engine_index]
            
            # (3) Generate Outputs Conditionally

            # SUMMARY
            try:
                if "Summary" in selected_options:
                    summarized_description = summary_engine["function"](summary_engine,summary_prompt_text,final_transcript)
                    (folder_path / "summary.txt").write_text(summarized_description, encoding="utf-8")
                else:
                    summarized_description = "[Summary was not generated]"
            except Exception as e:
                summarized_description = f"[Error generating summary: {e}]"
                logger.error(summarized_description)
                yield summarized_description, "", None

            # KEYWORDS
            try:
                if "Keywords" in selected_options:
                    seo_keywords = keywords_engine["function"](keywords_engine,keywords_prompt_text,final_transcript)
                    (folder_path / "keywords.txt").write_text(seo_keywords, encoding="utf-8")
                else:
                    seo_keywords = "[Keywords were not generated]"
            except Exception as e:
                seo_keywords = f"[Error generating keywords: {e}]"
                logger.error(seo_keywords)
                yield seo_keywords, "", None

            # TITLES
            try:
                if "Titles" in selected_options:
                    video_titles = titles_engine["function"](titles_engine,titles_prompt_text,final_transcript)
                    (folder_path / "titles.txt").write_text(video_titles, encoding="utf-8")
                else:
                    video_titles = "[Titles were not generated]"
            except Exception as e:
                video_titles = f"[Error generating titles: {e}]"
                logger.error(video_titles)
                yield video_titles, "", None

            # SHORTS
            try:
                if "Shorts" in selected_options:
                    shorts_sections = shorts_engine["function"](shorts_engine,shorts_prompt_text,final_transcript)
                    (folder_path / "shorts.txt").write_text(shorts_sections, encoding="utf-8")
                else:
                    shorts_sections = "[Shorts were not generated]"
            except Exception as e:
                shorts_sections = f"[Error generating shorts: {e}]"
                logger.error(shorts_sections)
                yield shorts_sections, "", None

            # (4) Build the HTML for the content of this tab
            content_html = f"""
            <h3>Transcript</h3>
            <pre>{final_transcript}</pre>
            {"<h3>Summary</h3><pre>" + summarized_description + "</pre>" if "Summary" in selected_options else ""}
            {"<h3>SEO Keywords</h3><pre>" + seo_keywords + "</pre>" if "Keywords" in selected_options else ""}
            {"<h3>Video Thumbnail Titles</h3><pre>" + video_titles + "</pre>" if "Titles" in selected_options else ""}
            {"<h3>Shorts</h3><pre>" + shorts_sections + "</pre>" if "Shorts" in selected_options else ""}
            <p><em>Outputs also saved in: {folder_path}</em></p>
            """

            # Add Radio + Label + Content to html_tabs
            checked = "checked" if i == 0 else ""
            html_tabs.append(f'<input type="radio" id="{tab_id}" name="tabs" {checked}>')
            html_tabs.append(f'<label for="{tab_id}">{Path(local_path).name}</label>')
            html_tabs.append(f'<div class="tab-content">{content_html}</div>')

    # Close the tab-buttons and tab-container divs after the loop
    html_tabs.append("</div>")  # close .tab-buttons
    html_tabs.append("</div>")  # close .tab-container

    # Zip the entire parent folder
    zip_path = shutil.make_archive(str(parent_folder), "zip", parent_folder)
    zip_file = Path(zip_path).absolute()
    logger.info(f"ZIP file created at: {zip_file}")

    # Clean up the parent folder after zipping
    shutil.rmtree(parent_folder)

    # Prepare the download link using Gradio's File component by returning the zip file path
    full_html = css_and_script + "\n".join(html_tabs)

    yield "Processing complete. Your files are ready for download below.", full_html, str(zip_file)

##############################
# Gradio App
##############################
with gr.Blocks(css=".footer.light {display: none !important;}", title="Podcast Audio Summarizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Podcast Audio Summarizer\n"
        "Upload any number of files, each file gets its own tab and every batch is "
        "saved out in structured folders as .txt files. You can also select which outputs "
        "you want generated and **edit** the prompt text for each option before processing."
    )
    with gr.Row():
        with gr.Column(scale=3):
            audio_input = gr.File(
                label="Upload Audio Files",
                file_count="multiple",
                type="filepath"
            )
        with gr.Column(scale=1):
            selected_options = gr.CheckboxGroup(
                choices=SELECTABLE_OPTIONS,
                value=["Summary", "Keywords", "Titles", "Shorts"],
                label="Select Desired Outputs"
            )
            transcription_engine = gr.Radio(
                choices = TRANSCRIPTION_OPTIONS,
                value = TRANSCRIPTION_OPTIONS[0],
                label = "Transcription Engine",
                interactive = True,
                type = "index"
            )


    # Under that row, we have four columns for the prompts
    with gr.Row():
        with gr.Column(min_width=240):
            summary_prompt_text = gr.Textbox(
                label="Summary Prompt",
                value=(
                    "Provide a three to four paragraph summary of this podcast episode transcript for a YouTube video description." 
                    "The summary should be crafted to appeal to both interested YouTube viewers and search engine algorithms like " 
                    "Google's and YouTube's crawlers. Extract and emphasize the key themes and topics from the transcript, ensuring " 
                    "the correct spelling of the hosts' names: Jordan Bloemen and Scott Francis Winder.\n\n"

                    "The tone should build intrigue and create a desire for viewers to watch the video, without revealing all the content." 
                    "Incorporate relevant keywords naturally within the summary to enhance SEO, focusing on terms that will help the video " 
                    "appear in search recommendations. Do not include a separate list of keywords or use markdown, bold text, or any " 
                    "additional formatting. Return the summary as plain text composed of well-structured paragraphs."
                    
                ),
                lines=16
            )

            add_prompt_refinement_modal(
                blocks=gr,
                openai_client=openai_client,
                main_prompt_textbox=summary_prompt_text,
                model="o1-mini",
                temperature=1,
                refine_button_label="Refine Prompt"
            )

            summary_prompt_engine = gr.Radio(
                choices = list(ENGINE_OPTIONS),
                value = list(ENGINE_OPTIONS)[0],
                label = "Summarization Model",
                interactive = True
            )
        with gr.Column(min_width=240):
            keywords_prompt_text = gr.Textbox(
                label="Keywords Prompt",
                value=(
                    "From the following transcript, provide a comma-separated list of unique primary keywords " 
                    "and keyword phrases that would improve SEO for YouTube videos. Ensure the total length of " 
                    "the comma-separated keywords is less than 500 characters. Focus on the main subjects and " 
                    "terms with high search volume and searchability. Analyze the entire transcript and include " 
                    "both single keywords and multi-word phrases without excluding essential keywords. Return the " 
                    "list as plain text with regular characters, separated by commas without any markdown or bold formatting."
                ),
                lines=16
            )

            add_prompt_refinement_modal(
                blocks=gr,
                openai_client=openai_client,
                main_prompt_textbox=keywords_prompt_text,
                model="o1-mini",
                temperature=1,
                refine_button_label="Refine Prompt"
            )
            
            keywords_prompt_engine = gr.Radio(
                choices = list(ENGINE_OPTIONS),
                value = list(ENGINE_OPTIONS)[0],
                label = "Summarization Model",
                interactive = True
            )
        with gr.Column(min_width=240):
            titles_prompt_text = gr.Textbox(
                label="Titles Prompt",
                value=(
                    "Using the transcript provided below, generate five video thumbnail title " 
                    "recommendations. Each title should be 5 to 6 words long, designed to build " 
                    "intrigue and encourage YouTube users to watch the video. Ensure the output is " 
                    "plain text only, without any markdown or bold formatting."
                ),
                lines=16
            )

            add_prompt_refinement_modal(
                blocks=gr,
                openai_client=openai_client,
                main_prompt_textbox=titles_prompt_text,
                model="o1-mini",
                temperature=1,
                refine_button_label="Refine Prompt"
            )

            titles_prompt_engine = gr.Radio(
                choices = list(ENGINE_OPTIONS),
                value = list(ENGINE_OPTIONS)[0],
                label = "Summarization Model",
                interactive = True
            )
        with gr.Column(min_width=240):
            shorts_prompt_text = gr.Textbox(
                label="Shorts Prompt",
                value=(
                    "Here is a transcript for a podcast episode. Please extract up to five of the highest " 
                    "quality sections that would make good 10-15 second shorts or clips for social sharing. " 
                    "The clips can be inspirational, humorous, or informative. Each clip should:"
                    "- Stand alone without requiring context from the complete episode"
                    "- Build intrigue to encourage viewers to engage with the full content"
                    "- Make a single point and be easily consumable on its own\n\n"
                    "Return the clips in plain text without any markdown, bolding, or formatting. " 
                    "Please list the clips in the order they appear chronologically within the transcript."
                ),
                lines=16
            )

            add_prompt_refinement_modal(
                blocks=gr,
                openai_client=openai_client,
                main_prompt_textbox=shorts_prompt_text,
                model="o1-mini",
                temperature=1,
                refine_button_label="Refine Prompt"
            )

            shorts_prompt_engine = gr.Radio(
                choices = list(ENGINE_OPTIONS),
                value = list(ENGINE_OPTIONS)[0],
                label = "Summarization Model",
                interactive = True
            )

    # Create a row with the "Process" button on top
    with gr.Row():
        submit_btn = gr.Button("Process", interactive=False)  # Button initialized as disabled

    # Enable the "Process" button after files are uploaded
    audio_input.change(
        fn=lambda files: gr.update(interactive=bool(files)),
        inputs=audio_input,
        outputs=submit_btn
    )

    status_box = gr.HTML(label="Status")
    html_output = gr.HTML()
    download_zip = gr.File(label="Download ZIP")  # Added Download ZIP component

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
            titles_prompt_text,
            shorts_prompt_text,
            transcription_engine,
            summary_prompt_engine,
            keywords_prompt_engine,
            titles_prompt_engine,
            shorts_prompt_engine
        ],
        outputs=[status_box, html_output, download_zip]  # Updated outputs to include download_zip
    ).then(
        fn=lambda: gr.update(interactive=True),
        inputs=None,
        outputs=submit_btn
    )

demo.launch(share=False, pwa=True)