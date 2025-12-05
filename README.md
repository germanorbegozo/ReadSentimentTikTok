# TikTok Sentiment Analysis

Analyze TikTok watch history and perform sentiment analysis on video content using AI.

For MacOS

1. Initialize the virtual environment and install the dependencies:

   ```bash
   uv venv
   ```
   
   ```bash
   source .venv/bin/activate
   ```

   ```bash
   uv pip install pip
   ```

   ```bash
   uv pip install -r requirements.txt
   ```

2. Set Gemini API key
    
    Open `process_watch_history.py` and set your API key at the top of the file:
    ```python
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
    ```
    
    **Get a free API key:** Sign up at [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)

3. **Run the analysis:**
   ```bash
   uv run process_watch_history.py
   ```

Requirements

- Python 3.8+
- FFmpeg
- Gemini API key (required - set in process_watch_history.py)
- TikTok watch history export in `data/Watch History.txt`

Output

Results are saved to `watch_history_analysis.csv`. Videos are downloaded to `downloads/` (skips already downloaded files).

