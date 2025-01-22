# Podcast Audio Summarizer
Using AI to facilitate the BS tasks in my life - Part I

A, mostly built by AI, Python-based tool that:

1.	**Splits** large audio files into ~25 MB chunks.
2.	**Transcribes** each chunk using OpenAI Whisper.
3.	Optionally uses ChatGPT to generate:
   •	**Summaries**
	•	**Keywords**
	•	**Titles**
4.	**Saves** each output in a structured folder and displays them in a Gradio UI with tabs.

## Features
-	**Large File Splitting** – Automatically splits audio into smaller segments to avoid 413 errors.
-	**Configurable Outputs** – Check/uncheck which outputs you want (Summary, Keywords, Titles).
-	**Editable Prompts** – Modify the text prompts for each output type before processing.
-	**Downloadable ZIP** – After processing, you can download a compressed folder of all TXT files.

## Requirements
-	Python 3.10+
-	See requirements.txt for package versions.

## Setup
1.	Clone or download this repository.
2.	Install dependencies:
```
pip install -r requirements.txt
```

3.	Create a .env file in the same directory (or set the environment variable some other way). For example:

```
OPENAI_API_KEY=sk-123abc...
```

4.	Install ffmpeg if you haven’t (pydub requires ffmpeg to process most audio formats).

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
5.	Optionally edit the prompts for Summary, Keywords, and Titles.
6.	Click “Process” to start transcription and GPT generation.
7.	Once complete, you can download the outputs as a ZIP via the link in the status field.

## Repository Structure
- podcast_summarizer.py – Main Python script with Gradio UI.
- requirements.txt – List of required Python packages.
- .env – (Not committed) Contains OPENAI_API_KEY or other secrets.
- .gitignore – (Recommended) to ignore .env and other local files.

## Contributing

Feel free to open issues or pull requests if you have improvements or suggestions.

## License

MIT License

# Happy summarizing!