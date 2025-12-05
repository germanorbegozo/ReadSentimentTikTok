#!/usr/bin/env python3

import os
import sys
import csv
import base64
import json
import time
from pathlib import Path
from datetime import datetime
import yt_dlp
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
import io

# ============================================================================
# GEMINI API CONFIGURATION
# ============================================================================
# Paste your Gemini API key here:
GEMINI_API_KEY = "AIzaSyDYmSsoAIRrQyQqs5a8iPimSqjFuIFEooE"
# Get your free API key at: https://makersuite.google.com/app/apikey
# ============================================================================

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: uv pip install google-generativeai")


def parse_watch_history(file_path):
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_date = None
    for line in lines:
        line = line.strip()
        if line.startswith('Date:'):
            date_str = line.replace('Date:', '').strip()
            try:
                current_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S UTC')
            except:
                current_date = None
        elif line.startswith('Link:'):
            url = line.replace('Link:', '').strip()
            if url and current_date:
                url = url.replace('tiktokv.com', 'tiktok.com')
                entries.append({'date': current_date, 'url': url})
    
    return entries


def download_video(url, output_dir='downloads', skip_if_exists=True, max_retries=3, use_cookies=False):
    """
    Download video with retry logic and optional cookie support.
    
    Args:
        url: Video URL to download
        output_dir: Directory to save video
        skip_if_exists: Skip if video already exists
        max_retries: Maximum number of retry attempts
        use_cookies: Whether to use browser cookies for authentication
    
    Returns:
        Tuple of (video_path, info_dict) or (None, None) if failed
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_template = os.path.join(output_dir, '%(id)s.%(ext)s')
    
    # Build yt-dlp options
    ydl_opts = {
        'outtmpl': output_template,
        'format': 'best',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': False,
    }
    
    # Add cookie support if requested
    if use_cookies:
        # Try to get cookies from browser automatically
        ydl_opts['cookiesfrombrowser'] = ('chrome',)  # Can also use 'firefox', 'edge', etc.
    
    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First, try to get video info
                info = ydl.extract_info(url, download=False)
                video_id = info.get('id', '')
                video_path = ydl.prepare_filename(info)
                
                if skip_if_exists and os.path.exists(video_path):
                    print(f"  Video already downloaded: {os.path.basename(video_path)}")
                    return video_path, info
                
                # Download the video
                ydl.download([url])
                
                # Verify file exists
                if os.path.exists(video_path):
                    return video_path, info
                else:
                    raise Exception("Download completed but file not found")
                    
        except yt_dlp.utils.DownloadError as e:
            error_str = str(e)
            
            # Check for specific error types
            if 'IP address is blocked' in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                    print(f"  IP blocked, retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  Error: IP address blocked. Consider using cookies or VPN.")
                    return None, None
                    
            elif 'Log in for access' in error_str or 'cookies' in error_str.lower():
                if not use_cookies and attempt == 0:
                    print(f"  Warning: Video requires authentication. Consider using cookies.")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"  Authentication required, retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  Error: Video requires login. Use --cookies-from-browser or set use_cookies=True")
                    return None, None
                    
            else:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  Download failed, retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  Error downloading: {error_str[:100]}")
                    return None, None
                    
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  Unexpected error, retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  Error downloading {url}: {str(e)[:100]}")
                return None, None
    
    return None, None


def analyze_with_llm(transcription, ocr_text, screenshots):
    """
    Analyze video content using Google's Gemini Pro Vision model.
    Returns structured analysis including description, sentiment, and image description.
    """
    if not GEMINI_AVAILABLE:
        print("  Error: google-generativeai not installed. Install with: uv pip install google-generativeai")
        return None
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("  Error: GEMINI_API_KEY not set. Please set it at the top of the script.")
        return None
    
    try:
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        
        # List available models and find one that supports vision
        try:
            available_models = list(genai.list_models())
            print(f"  Available models: {[m.name for m in available_models[:5]]}")
            
            # Look for models that support generateContent (vision capable)
            vision_model = None
            # Prefer pro models over flash for better quality
            for m in available_models:
                if 'generateContent' in m.supported_generation_methods:
                    model_name = m.name.split('/')[-1]
                    # Prefer pro models, then flash
                    if 'pro' in model_name.lower():
                        vision_model = model_name
                        print(f"  Using model: {vision_model}")
                        break
            
            # If no pro model, use flash
            if not vision_model:
                for m in available_models:
                    if 'generateContent' in m.supported_generation_methods:
                        model_name = m.name.split('/')[-1]
                        if 'flash' in model_name.lower():
                            vision_model = model_name
                            print(f"  Using model: {vision_model}")
                            break
            
            if not vision_model:
                # Fallback: try common model names
                for name in ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro']:
                    try:
                        test_model = genai.GenerativeModel(name)
                        vision_model = name
                        print(f"  Using model: {vision_model}")
                        break
                    except:
                        continue
            
            if not vision_model:
                raise Exception("No suitable Gemini model found")
            
            model = genai.GenerativeModel(vision_model)
        except Exception as e:
            print(f"  Warning: Could not list models: {e}")
            # Try gemini-1.5-flash as default (most commonly available)
            print("  Trying gemini-1.5-flash as default...")
            model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare text content
        text_content = ""
        if transcription and transcription.strip():
            text_content += f"Audio Transcription: {transcription[:2000]}\n\n"
        if ocr_text and ocr_text.strip():
            text_content += f"Text extracted from video frames: {ocr_text[:1000]}\n\n"
        
        # Create the prompt
        prompt = """Analyze this TikTok video content and provide a comprehensive analysis in JSON format.

Consider:
1. The audio transcription (what was said)
2. Any text visible in the video frames
3. The visual content in the screenshots

Provide your analysis as a JSON object with the following structure:
{
  "description": "A one-sentence description of what the video is about",
  "sentiment_analysis": {
    "sentiment_label": "positive, negative, or neutral",
    "sentiment_score": A number between -1.0 (very negative) and 1.0 (very positive),
    "confidence": A number between 0.0 and 1.0 indicating confidence in the sentiment analysis,
    "reasoning": "Brief explanation of why this sentiment was assigned"
  },
  "image_description": "A brief description of the visual content shown in the video screenshots, including any notable elements, people, objects, scenes, or activities visible"
}

Be thorough and accurate in your analysis. Return ONLY valid JSON, no other text."""
        
        # Prepare content parts for Gemini
        # For gemini-pro-vision, combine text and images in the content list
        content_parts = []
        
        # Start with prompt
        if text_content:
            full_prompt = prompt + "\n\n" + text_content
        else:
            full_prompt = prompt
        
        content_parts.append(full_prompt)
        
        # Add screenshots (up to 5)
        screenshot_count = 0
        for i, screenshot in enumerate(screenshots[:5]):
            try:
                content_parts.append(screenshot)
                screenshot_count += 1
            except Exception as e:
                print(f"  Warning: Could not add screenshot {i}: {e}")
        
        # If we have no screenshots and no text, can't analyze
        if screenshot_count == 0 and not text_content:
            print("  Warning: No screenshots and no transcription text available for LLM analysis")
            return None
        
        # Call Gemini API with safety settings configured
        print(f"  Calling Gemini API with {screenshot_count} screenshots...")
        
        # Configure safety settings to be more permissive (needed for video content analysis)
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
        
        try:
            response = model.generate_content(
                content_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2000,  # Increased for longer responses
                ),
                safety_settings=safety_settings
            )
        except Exception as e:
            error_str = str(e)
            if "finish_reason" in error_str or "2" in error_str:
                print("  Warning: Content was blocked by safety filters. Trying with text-only analysis...")
                # Retry with text only (no images)
                if text_content:
                    try:
                        text_only_prompt = prompt + "\n\n" + text_content
                        response = model.generate_content(
                            text_only_prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.3,
                                max_output_tokens=2000,
                            ),
                            safety_settings=safety_settings
                        )
                    except Exception as e2:
                        raise Exception(f"Text-only analysis also failed: {e2}")
                else:
                    raise Exception("Content blocked and no text available for analysis")
            else:
                raise
        
        # Check finish reason
        if hasattr(response, 'candidates') and response.candidates:
            finish_reason = response.candidates[0].finish_reason
            if finish_reason == 2:  # SAFETY
                print("  Warning: Response was blocked by safety filters")
                raise Exception("Content blocked by safety filters")
            elif finish_reason == 3:  # RECITATION
                print("  Warning: Response was blocked due to recitation")
                raise Exception("Content blocked due to recitation")
        
        # Extract JSON from response
        try:
            result_text = response.text.strip()
        except Exception as e:
            if "finish_reason" in str(e) or "Part" in str(e):
                print("  Error: Response was blocked or incomplete")
                raise Exception("Response was blocked or incomplete - try with different content")
            raise
        
        # Try to extract JSON if wrapped in markdown code blocks
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        # Handle truncated JSON (common issue)
        if not result_text.endswith('}'):
            # Try to find the last complete JSON object
            last_brace = result_text.rfind('}')
            if last_brace > 0:
                result_text = result_text[:last_brace + 1]
            else:
                # If no closing brace, try to fix common truncation issues
                if '"description"' in result_text and '"sentiment_analysis"' in result_text:
                    # Try to complete the JSON
                    if '"image_description"' not in result_text:
                        result_text = result_text.rstrip().rstrip(',') + ',\n  "image_description": "Analysis incomplete due to response truncation"\n}'
                    else:
                        result_text = result_text.rstrip().rstrip(',') + '\n}'
        
        try:
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError as e:
            print(f"  Warning: JSON parsing failed. Response length: {len(result_text)}")
            print(f"  Response preview: {result_text[:200]}...")
            # Try to extract partial data if possible
            raise Exception(f"Failed to parse JSON response: {e}")
        
    except json.JSONDecodeError as e:
        print(f"  Error: Failed to parse Gemini response as JSON: {e}")
        print(f"  Response was: {response.text[:200] if 'response' in locals() else 'No response'}")
        return None
    except Exception as e:
        error_msg = str(e)
        print(f"  Error: Gemini API call failed: {error_msg}")
        if "API_KEY_INVALID" in error_msg or "401" in error_msg:
            print("  Error: Invalid API key. Please check your GEMINI_API_KEY.")
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            print("  Error: Rate limit exceeded. Please wait and try again.")
        elif "quota" in error_msg.lower():
            print("  Error: Quota exceeded. Please check your Gemini API quota.")
        return None


def extract_screenshots(video_path, num_frames=5):
    """
    Extract screenshots from video at evenly spaced intervals.
    Returns list of PIL Image objects.
    """
    screenshots = []
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        
        timestamps = []
        if num_frames == 1:
            timestamps = [duration / 2]
        else:
            for i in range(num_frames):
                if i == 0:
                    timestamp = duration * 0.1  # 10% in (avoid very start)
                elif i == num_frames - 1:
                    timestamp = duration * 0.9  # 90% in (avoid very end)
                else:
                    timestamp = (i / (num_frames - 1)) * duration * 0.8 + duration * 0.1
                timestamps.append(timestamp)
        
        for timestamp in timestamps:
            try:
                frame = video.get_frame(timestamp)
                img = Image.fromarray(frame)
                screenshots.append(img)
            except Exception as e:
                print(f"  Warning: Could not extract frame at {timestamp:.2f}s: {e}")
        
        video.close()
    except Exception as e:
        print(f"  Warning: Could not extract screenshots: {e}")
    
    return screenshots


def image_to_base64(image):
    """
    Convert PIL Image to base64 string for API transmission.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def extract_text_from_frames(video_path, num_frames=5):
    if not OCR_AVAILABLE:
        return ""
    
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        extracted_text = []
        
        for i in range(num_frames):
            timestamp = (i + 0.5) * duration / num_frames
            frame = video.get_frame(timestamp)
            img = Image.fromarray(frame)
            
            try:
                text = pytesseract.image_to_string(img, lang='eng')
                if text.strip():
                    extracted_text.append(text.strip())
            except:
                pass
        
        video.close()
        return ' '.join(extracted_text)
    except Exception as e:
        return ""


def analyze_video(video_path, whisper_model_instance):
    audio_path = str(Path(video_path).with_suffix('.wav'))
    
    try:
        video = VideoFileClip(video_path)
        duration = video.duration if hasattr(video, 'duration') else 0
        
        # Extract transcription
        transcription = ""
        language = 'unknown'
        
        if video.audio is not None:
            video.audio.write_audiofile(audio_path, logger=None)
            result = whisper_model_instance.transcribe(audio_path)
            transcription = result['text']
            language = result.get('language', 'unknown')
            os.remove(audio_path)
        
        # Extract OCR text from frames
        ocr_text = extract_text_from_frames(video_path)
        
        # If no transcription, use OCR text
        if not transcription.strip() and ocr_text:
            transcription = ocr_text
            language = 'unknown'
        
        # Extract screenshots for LLM analysis
        screenshots = extract_screenshots(video_path, num_frames=5)
        
        video.close()
        
        # Get modal words (top 5 most frequent words)
        text_lower = transcription.lower() if transcription else ""
        words = [w for w in text_lower.split() if len(w) > 4]
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        topics_list = [word for word, freq in topics]
        
        # Pad to 5 words with empty strings
        while len(topics_list) < 5:
            topics_list.append('')
        
        # Analyze using Gemini LLM (required - no fallback)
        llm_analysis = analyze_with_llm(transcription, ocr_text, screenshots)
        
        if not llm_analysis:
            print("  Error: LLM analysis failed. Cannot proceed without Gemini API.")
            return None
        
        # Extract LLM analysis results
        description = llm_analysis.get('description', '')
        sentiment_data = llm_analysis.get('sentiment_analysis', {})
        image_description = llm_analysis.get('image_description', '')
        
        sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
        sentiment_score = sentiment_data.get('sentiment_score', 0.0)
        sentiment_confidence = sentiment_data.get('confidence', 0.0)
        sentiment_reasoning = sentiment_data.get('reasoning', '')
        
        return {
            'transcription': transcription,
            'language': language,
            'duration': duration,
            'description': description,
            'sentiment_label': sentiment_label,
            'sentiment_score': sentiment_score,
            'sentiment_confidence': sentiment_confidence,
            'sentiment_reasoning': sentiment_reasoning,
            'image_description': image_description,
            'modal_word_1': topics_list[0] if len(topics_list) > 0 else '',
            'modal_word_2': topics_list[1] if len(topics_list) > 1 else '',
            'modal_word_3': topics_list[2] if len(topics_list) > 2 else '',
            'modal_word_4': topics_list[3] if len(topics_list) > 3 else '',
            'modal_word_5': topics_list[4] if len(topics_list) > 4 else '',
            'word_count': len(transcription.split()) if transcription else 0
        }
        
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        return None


def get_processed_urls(csv_path):
    processed_urls = set()
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('url'):
                        processed_urls.add(row['url'])
        except:
            pass
    return processed_urls


def main():
    history_file = 'data/Watch History.txt'
    
    if not os.path.exists(history_file):
        print(f"Error: {history_file} not found")
        sys.exit(1)
    
    # Check for Gemini API key (required)
    if not GEMINI_AVAILABLE:
        print("\n" + "="*70)
        print("ERROR: google-generativeai not installed!")
        print("="*70)
        print("Install with: uv pip install google-generativeai")
        print("="*70 + "\n")
        sys.exit(1)
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("\n" + "="*70)
        print("ERROR: GEMINI_API_KEY not set!")
        print("="*70)
        print("Please set your Gemini API key at the top of process_watch_history.py")
        print("Get your free API key at: https://makersuite.google.com/app/apikey")
        print("="*70 + "\n")
        sys.exit(1)
    
    print(f"✓ Gemini API key configured - using Gemini Pro for analysis\n")
    
    print("Parsing watch history...")
    entries = parse_watch_history(history_file)
    print(f"Found {len(entries)} TikTok videos in watch history")
    
    csv_path = 'watch_history_analysis.csv'
    processed_urls = get_processed_urls(csv_path)
    
    if processed_urls:
        print(f"\nFound existing CSV with {len(processed_urls)} processed videos")
        resume = input("Resume from where you left off? (y/n) [default: y]: ").strip().lower() or "y"
        if resume == 'y':
            entries = [e for e in entries if e['url'] not in processed_urls]
            print(f"Resuming: {len(entries)} videos remaining to process")
    else:
        resume = "n"
    
    if not entries:
        print("\nAll videos already processed!")
        sys.exit(0)
    
    limit_input = input(f"\nProcess only first N videos? (Enter number or 'all' for all {len(entries)}) [default: all]: ").strip()
    if limit_input.lower() == 'all' or limit_input == '':
        limit = len(entries)
    else:
        try:
            limit = int(limit_input)
            if limit > len(entries):
                limit = len(entries)
        except:
            limit = len(entries)
    
    entries = entries[:limit]
    print(f"Processing {len(entries)} videos")
    
    skip_downloaded = input("\nSkip already-downloaded videos? (y/n) [default: y]: ").strip().lower() or "y"
    skip_downloaded = skip_downloaded == 'y'
    
    use_cookies = input("\nUse browser cookies for authentication? (helps with blocked/private videos) (y/n) [default: n]: ").strip().lower() or "n"
    use_cookies = use_cookies == 'y'
    if use_cookies:
        print("  Note: Will attempt to use cookies from Chrome browser")
    
    print("\nWhisper model:")
    print("  tiny   = fastest (less accurate)")
    print("  base   = balanced (recommended)")
    print("  small  = slower (more accurate)")
    model = input("Choose [default: tiny]: ").strip() or "tiny"
    
    print(f"\nLoading Whisper model ({model})...")
    whisper_model = whisper.load_model(model)
    
    download_dir = 'downloads'
    
    print(f"\nProcessing {len(entries)} videos...")
    print("="*70)
    
    file_exists = os.path.exists(csv_path)
    
    for i, entry in enumerate(entries, 1):
        date = entry['date']
        url = entry['url']
        
        print(f"\n[{i}/{len(entries)}] Processing video watched on {date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"URL: {url}")
        
        video_path, info = download_video(url, download_dir, skip_if_exists=skip_downloaded, use_cookies=use_cookies)
        
        if not video_path:
            print("  Skipping - download failed after retries")
            # Still write a record with minimal info
            result = {
                'watch_date': date.isoformat(),
                'url': url,
                'video_id': '',
                'video_description': '',
                'creator': '',
                'duration_seconds': 0,
                'language': 'unknown',
                'transcription': '',
                'description': 'Download failed - video unavailable or requires authentication',
                'image_description': '',
                'modal_word_1': '',
                'modal_word_2': '',
                'modal_word_3': '',
                'modal_word_4': '',
                'modal_word_5': '',
                'sentiment_label': 'unknown',
                'sentiment_score': 0.0,
                'sentiment_confidence': 0.0,
                'sentiment_reasoning': 'Download failed - unable to analyze',
            }
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                fieldnames = list(result.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                writer.writerow(result)
            continue
        
        print("  Analyzing...")
        analysis = analyze_video(video_path, whisper_model)
        
        if not analysis:
            print("  Skipping - analysis failed")
            continue
        
        video_id = info.get('id', '') if info else Path(video_path).stem
        video_description = info.get('description', '') if info else ''
        uploader = info.get('uploader', '') if info else ''
        
        result = {
            'watch_date': date.isoformat(),
            'url': url,
            'video_id': video_id,
            'video_description': video_description,
            'creator': uploader,
            'duration_seconds': round(analysis['duration'], 2),
            'language': analysis['language'],
            'transcription': analysis['transcription'],
            'description': analysis['description'],
            'image_description': analysis['image_description'],
            'modal_word_1': analysis['modal_word_1'],
            'modal_word_2': analysis['modal_word_2'],
            'modal_word_3': analysis['modal_word_3'],
            'modal_word_4': analysis['modal_word_4'],
            'modal_word_5': analysis['modal_word_5'],
            'sentiment_label': analysis['sentiment_label'],
            'sentiment_score': analysis['sentiment_score'],
            'sentiment_confidence': analysis['sentiment_confidence'],
            'sentiment_reasoning': analysis['sentiment_reasoning'],
        }
        
        print(f"  Sentiment: {analysis['sentiment_label']} (score: {analysis['sentiment_score']:.2f})")
        print(f"  Description: {analysis['description'][:60]}...")
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = list(result.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
                file_exists = True
            writer.writerow(result)
        
        if i % 10 == 0:
            print(f"\nProgress: {i}/{len(entries)} videos processed (saved to {csv_path})")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"Results saved to: {csv_path}")
    print(f"Videos processed in this session: {len(entries)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        csv_path = 'watch_history_analysis.csv'
        print(f"\n\nProcessing cancelled. Progress saved to {csv_path}")
        sys.exit(0)

