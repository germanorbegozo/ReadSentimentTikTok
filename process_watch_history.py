#!/usr/bin/env python3

import os
import sys
import csv
import re
from pathlib import Path
from datetime import datetime
import yt_dlp
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


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


def download_video(url, output_dir='downloads', skip_if_exists=True):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_template = os.path.join(output_dir, '%(id)s.%(ext)s')
    
    ydl_opts = {
        'outtmpl': output_template,
        'format': 'best',
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_id = info.get('id', '')
            video_path = ydl.prepare_filename(info)
            
            if skip_if_exists and os.path.exists(video_path):
                print(f"  Video already downloaded: {os.path.basename(video_path)}")
                return video_path, info
            
            ydl.download([url])
            return video_path, info
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None, None


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
        
        if video.audio is None:
            video_text = extract_text_from_frames(video_path)
            video.close()
            
            if video_text:
                text_lower = video_text.lower()
                words = [w for w in text_lower.split() if len(w) > 4]
                word_freq = {}
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
                topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
                topics_list = [word for word, freq in topics]
                
                if len(video_text) > 200:
                    context_summary = f"Video displays text content: {video_text[:197]}..."
                else:
                    context_summary = f"Video displays text: {video_text}."
                
                return {
                    'transcription': video_text,
                    'language': 'unknown',
                    'duration': duration,
                    'sentiment': 'neutral',
                    'topics': ', '.join(topics_list),
                    'context': context_summary
                }
            else:
                return {
                    'transcription': '',
                    'language': 'none',
                    'duration': duration,
                    'sentiment': 'neutral',
                    'topics': '',
                    'context': 'Silent video with no spoken content or visible text.'
                }
        
        video.audio.write_audiofile(audio_path, logger=None)
        video.close()
        
        result = whisper_model_instance.transcribe(audio_path)
        transcription = result['text']
        language = result.get('language', 'unknown')
        
        os.remove(audio_path)
        
        if not transcription.strip():
            video_text = extract_text_from_frames(video_path)
            if video_text:
                transcription = video_text
                language = 'unknown'
        
        text_lower = transcription.lower()
        
        positive_words = ['love', 'great', 'amazing', 'awesome', 'happy', 'best', 'wonderful', 'good', 'beautiful']
        negative_words = ['hate', 'bad', 'terrible', 'awful', 'sad', 'worst', 'horrible', 'wrong', 'ugly']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'positive'
        elif neg_count > pos_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        words = [w for w in text_lower.split() if len(w) > 4]
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        topics_list = [word for word, freq in topics]
        
        clean_text = transcription.strip()
        text_lower = clean_text.lower()
        
        if not clean_text:
            context_summary = "Silent video with no spoken content or visible text."
        else:
            sentences = [s.strip() for s in clean_text.split('.') if s.strip() and len(s.strip()) > 10]
            
            if not sentences:
                context_summary = clean_text[:150] + ('...' if len(clean_text) > 150 else '.')
            else:
                first_sentence = sentences[0]
                
                if len(sentences) == 1:
                    if len(first_sentence) > 200:
                        context_summary = first_sentence[:197] + '...'
                    else:
                        context_summary = first_sentence + ('.' if not first_sentence.endswith('.') else '')
                else:
                    if 'dating' in text_lower or 'relationship' in text_lower:
                        if 'standards' in text_lower or 'picky' in text_lower:
                            context_summary = "Creator discusses their dating experiences and how their personal growth has raised their standards, making it harder to find compatible partners."
                        else:
                            context_summary = "Creator shares thoughts and experiences about dating and relationships."
                    elif 'coffee' in text_lower and 'friend' in text_lower:
                        context_summary = "Creator recounts a conversation with a friend, sharing personal reflections and insights."
                    elif 'tutorial' in text_lower or 'how to' in text_lower or 'teach' in text_lower:
                        if 'how to' in text_lower:
                            idx = text_lower.find('how to')
                            topic = clean_text[idx:idx+80].split('.')[0].strip()
                            context_summary = f"Educational content explaining {topic}."
                        else:
                            context_summary = "Tutorial or educational content providing instructions or guidance."
                    elif 'roast' in text_lower and ('gpt' in text_lower or 'chat' in text_lower):
                        context_summary = "Creator uses AI to generate a humorous roast about themselves or someone else."
                    elif 'son' in text_lower and 'girlfriend' in text_lower:
                        context_summary = "Parent observes and discusses their son's potential romantic relationship."
                    elif 'communication' in text_lower or 'communicate' in text_lower:
                        if 'pareja' in text_lower or 'relationship' in text_lower:
                            context_summary = "Advice about improving communication in romantic relationships."
                        else:
                            context_summary = "Content focused on communication skills and interpersonal relationships."
                    elif 'story' in text_lower or 'happened' in text_lower or 'remember' in text_lower:
                        context_summary = "Personal storytelling sharing an experience or memory."
                    elif 'dance' in text_lower or 'dancing' in text_lower:
                        context_summary = "Dance performance or dance-related content."
                    elif 'music' in text_lower or 'song' in text_lower:
                        context_summary = "Music-related content or performance."
                    elif 'review' in text_lower or 'rating' in text_lower:
                        context_summary = "Review or opinion about a product, service, or content."
                    else:
                        if len(first_sentence) > 150:
                            words = first_sentence.split()
                            summary_words = []
                            char_count = 0
                            for word in words:
                                if char_count + len(word) + 1 > 150:
                                    break
                                summary_words.append(word)
                                char_count += len(word) + 1
                            context_summary = ' '.join(summary_words) + '...'
                        else:
                            context_summary = first_sentence + ('.' if not first_sentence.endswith('.') else '')
        
        return {
            'transcription': transcription,
            'language': language,
            'duration': duration,
            'sentiment': sentiment,
            'topics': ', '.join(topics_list),
            'context': context_summary,
            'word_count': len(transcription.split())
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
        
        video_path, info = download_video(url, download_dir, skip_if_exists=skip_downloaded)
        
        if not video_path:
            print("  Skipping - download failed")
            continue
        
        print("  Analyzing...")
        analysis = analyze_video(video_path, whisper_model)
        
        if not analysis:
            print("  Skipping - analysis failed")
            continue
        
        video_id = info.get('id', '') if info else Path(video_path).stem
        description = info.get('description', '') if info else ''
        uploader = info.get('uploader', '') if info else ''
        
        result = {
            'watch_date': date.isoformat(),
            'url': url,
            'video_id': video_id,
            'description': description,
            'creator': uploader,
            'duration_seconds': round(analysis['duration'], 2),
            'language': analysis['language'],
            'transcription': analysis['transcription'],
            'context': analysis['context'],
            'topics': analysis['topics'],
            'sentiment': analysis['sentiment'],
        }
        
        print(f"  Sentiment: {analysis['sentiment']}")
        print(f"  Context: {analysis['context'][:60]}...")
        
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

