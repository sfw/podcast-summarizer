# Podcast Audio Summarizer
Using AI to facilitate the BS tasks in my life - Part I

A, mostly built by AI, Python-based tool that:

1.	**Splits** large audio files into ~25 MB chunks.
2.	**Transcribes** each chunk using AssemblyAI for speaker identification but the code is commented out to use OpenAI whisper if you'd like.
3.	Optionally uses an AI model of your choice to generate:
	- **Summaries**
	- **Keywords**
	- **Titles**
 	- **Short Clips** 
4.	**Saves** each output in a structured folder and displays them in a Gradio UI with tabs.

## Features
-	**Large File Splitting** – Automatically splits audio into smaller segments to avoid 413 errors.
-	**Configurable Outputs** – Check/uncheck which outputs you want (Summary, Keywords, Titles).
-	**Editable Prompts** – Modify the text prompts for each output type before processing.
-	**Downloadable ZIP** – After processing, you can download a compressed folder of all TXT files.

## Requirements
-	Python 3.10+
-	See requirements.txt for package versions.
-	OpenAI API account + credits
-	Assembly API account + credits (optional)
-	Google Gemini API account + credits (optional)

## Setup
1.	Clone or download this repository.
2.	Install dependencies:
```
pip install -r requirements.txt
```

3.	Create a .env file in the same directory (or set the environment variable some other way). For example:

```
OPENAI_API_KEY=sk-123abc...
ASSEMBLYAI_API_KEY=12341234...
GEMINI_API_KEY=AIabc...
DEEPSEEK_API_KEY=
```

4.	Install ffmpeg if you haven’t (pydub requires ffmpeg to process most audio formats).
5.	Update default OpenAI prompts

```
       with gr.Column(min_width=240):
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
        with gr.Column(min_width=240):
            keywords_prompt_text = gr.Textbox(
                label="Keywords Prompt",
                value=(
                    "From the following transcript, provide a comma-separated list of top relevant keywords "
                    "that would improve SEO. Focus on main subjects and terms. Return plain text, no markdown, no bold text."
                ),
                lines=8
            )
        with gr.Column(min_width=240):
            titles_prompt_text = gr.Textbox(
                label="Titles Prompt",
                value=(
                    "Provide five short video thumbnail title recommendations (max 6 words each) "
                    "from this transcript. Return plain text, no markdown, no bold text."
                ),
                lines=8
            )
        with gr.Column(min_width=240):
            shorts_prompt_text = gr.Textbox(
                label="Shorts Prompt",
                value=(
                    "Here is a transcript for a podcast episode. Can you pull out three sections of it that "
                    "would make good 10-15 second shorts or clips for social sharing. Focus on content that "
                    "will be understandable without the context of the complete episode but also interesting."
                ),
                lines=8
            )
```
6. Define any other AI model you prefer (currently OpenAI/DeepSeek and Google Models are supported)

## Usage
1.	Run the script:
```
python podcast-summarizer.py
```

2.	You’ll see console output like:
```
Running on local URL:  http://127.0.0.1:7860
```
3.	Open the link in your browser to access the Gradio interface.
4.	Upload one or more audio files.
5.	Select a transcription engine.
6.	Optionally, edit the prompts for Summary, Keywords, and Titles.
7.	Optionally, select the engine/model you'd prefer for each prompt.
8.	Click “Process” to start transcription and GPT generation.
9.	Once complete, you can download the outputs as a ZIP via the link.

## Repository Structure
- podcast_summarizer.py – Main Python script with Gradio UI.
- requirements.txt – List of required Python packages.
- .env – (Not committed) Contains OPENAI_API_KEY or other secrets.
- .gitignore – (Recommended) to ignore .env and other local files.

## Contributing

Feel free to open issues or pull requests if you have improvements or suggestions.

## License

MIT License

# Acknowledgements
- [Gradio](https://gradio.app/) for the user-friendly interface.
- [OpenAI](https://openai.com/) for the Whisper and ChatGPT APIs.
- [Pydub](https://github.com/jiaaro/pydub) for audio processing.
- [AssemblyAI](https://www.assemblyai.com/) for audio transcription with speaker ID.
- [Google Genai](https://ai.google.dev/gemini-api/docs/sdks) for Google Gemini Models.

# Happy summarizing!
