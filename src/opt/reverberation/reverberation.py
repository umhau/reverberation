#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
import threading
import queue
import subprocess
import sys
import os
import pyaudio
import wave
import tempfile
import time
from faster_whisper import WhisperModel
import numpy as np
import logging
from datetime import datetime
import json
from pathlib import Path

# Set up logging
log_file = f"/tmp/whisper_transcribe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Starting whisper-transcribe, log file: {log_file}")

# Model configurations
# Note: int8 is the quantized version for CPU, float16 requires CUDA
# Available compute types: int8 (CPU), float16 (GPU), float32 (both)
MODELS = [
    {"name": "tiny.en", "model": "tiny.en", "device": "cpu", "compute_type": "int8"},
    {"name": "base.en", "model": "base.en", "device": "cpu", "compute_type": "int8"},
    {"name": "small.en", "model": "small.en", "device": "cpu", "compute_type": "int8"}
]

# Config file path
CONFIG_PATH = Path.home() / ".config" / "reverberation" / "config.json"

def load_config():
    """Load configuration from file"""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"model_index": 0}

def save_config(config):
    """Save configuration to file"""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f)

class TranscribeWindow:
    def __init__(self):
        logger.info("Initializing TranscribeWindow")
        self.root = tk.Tk()
        
        # Hide window initially to prevent flash
        self.root.withdraw()
        
        self.root.title("reverberation")
        logger.info("Created Tk root window")
        
        # Remove window decorations and make it stay on top
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        
        # Set window type for i3 to treat it as floating
        self.root.wm_attributes('-type', 'dialog')
        
        # Style configuration (dmenu-like)
        self.bg_color = "#222222"
        self.fg_color = "#eeeeee"
        self.highlight_color = "#005577"
        
        # Set window size and center it (increased height for model selection)
        window_width = 600
        window_height = 500
        
        # Update window first to get accurate screen dimensions
        self.root.update_idletasks()
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate position
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg=self.bg_color)
        
        # Create main frame
        self.main_frame = tk.Frame(self.root, bg=self.bg_color)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status label
        self.status_label = tk.Label(
            self.main_frame,
            text="loading tiny.en .",
            bg=self.bg_color,
            fg=self.fg_color,
            font=("monospace", 10)
        )
        self.status_label.pack(pady=5)
        
        # Buffer indicator frame
        self.buffer_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.buffer_frame.pack(pady=2)
        
        # Create 9 dots for buffer visualization
        self.buffer_dots = []
        for i in range(9):
            dot = tk.Label(
                self.buffer_frame,
                text="○",  # Empty circle
                bg=self.bg_color,
                fg="#444444",  # Dark gray for empty
                font=("monospace", 12)
            )
            dot.pack(side=tk.LEFT, padx=1)
            self.buffer_dots.append(dot)
        
        # Buffer tracking variables
        self.buffer_progress = 0
        self.is_processing = False
        
        # Help text (pack at bottom first)
        self.help_label = tk.Label(
            self.main_frame,
            text="[Tab] Switch model  |  [ESC] Cancel  |  [Enter] Insert text  |  [Shift+Enter] Copy to clipboard",
            bg=self.bg_color,
            fg="#888888",
            font=("monospace", 9)
        )
        self.help_label.pack(side=tk.BOTTOM, pady=2)
        
        # Model selection frame (pack at bottom second)
        self.model_frame = tk.Frame(self.main_frame, bg="#333333", height=50)  # Different bg to see it
        self.model_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=8)
        self.model_frame.pack_propagate(False)  # Maintain fixed height
        
        # Load saved config
        self.config = load_config()
        self.current_model_index = self.config.get("model_index", 0)
        
        # Create model labels with better visibility
        self.model_labels = []
        for i, model in enumerate(MODELS):
            label = tk.Label(
                self.model_frame,
                text=f" {model['name']} ",
                bg=self.bg_color,
                fg=self.fg_color,
                font=("monospace", 11, "bold"),
                padx=15,
                pady=5,
                relief="solid",
                borderwidth=1
            )
            label.pack(side=tk.LEFT, padx=8)
            self.model_labels.append(label)
        
        # Update model highlighting
        self.update_model_highlight()
        
        # Text display (pack after bottom elements are in place)
        self.text_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        self.text_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 5))
        
        self.text_display = tk.Text(
            self.text_frame,
            bg=self.bg_color,
            fg=self.fg_color,
            font=("monospace", 12),
            wrap=tk.WORD,
            insertbackground=self.fg_color,
            selectbackground=self.highlight_color,
            selectforeground=self.fg_color,
            borderwidth=0,
            highlightthickness=0
        )
        self.text_display.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(self.text_frame, command=self.text_display.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_display.config(yscrollcommand=scrollbar.set)
        
        # Bind keys - use root window and all children
        logger.info("Setting up key bindings")
        self.root.bind_all('<Escape>', self.on_escape)
        self.root.bind_all('<Return>', self.on_return)
        self.root.bind_all('<Shift-Return>', self.on_shift_return)
        self.root.bind_all('<Tab>', self.on_tab)
        
        # Also bind to text display specifically
        self.text_display.bind('<Escape>', self.on_escape)
        self.text_display.bind('<Return>', self.on_return)
        self.text_display.bind('<Shift-Return>', self.on_shift_return)
        self.text_display.bind('<Tab>', self.on_tab)
        
        # Debug: print when keys are pressed
        self.root.bind_all('<Key>', self.on_any_key)
        
        # We'll set focus and grab after window is mapped
        self.root.after(10, self.setup_focus_and_grab)
        
        # Audio and model setup
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.is_recording = False
        self.model = None
        self.audio_thread = None
        self.transcribe_thread = None
        self.loading_dots = 1
        self.is_loading = True
        self.is_reloading = False
        
        # Start loading model in background
        threading.Thread(target=self.load_model, daemon=True).start()
        
        # Update UI periodically
        self.update_ui()
        
        # Start loading animation
        self.animate_loading()
        
    def setup_focus_and_grab(self):
        """Set up focus and keyboard grab after window is mapped"""
        logger.info("Setting up focus and keyboard grab")
        
        # First ensure window is visible and mapped
        self.root.update_idletasks()
        self.root.lift()
        
        # Set focus
        self.root.focus_force()
        self.text_display.focus_set()
        
        # Wait a bit more then grab keyboard
        self.root.after(100, self.grab_keyboard)
        
    def grab_keyboard(self):
        """Grab keyboard input exclusively"""
        try:
            logger.info("Attempting to grab keyboard")
            self.root.grab_set()
            self.root.grab_set_global()  # This grabs ALL keyboard input
            logger.info(f"Keyboard grabbed successfully")
            logger.info(f"Grab current: {self.root.grab_current()}")
            logger.info(f"Focus: {self.root.focus_get()}")
        except Exception as e:
            logger.error(f"Failed to grab keyboard: {e}")
        
    def update_model_highlight(self):
        """Update model label highlighting"""
        for i, label in enumerate(self.model_labels):
            if i == self.current_model_index:
                # Highlight selected model
                label.config(
                    bg=self.highlight_color,
                    fg=self.bg_color,
                    relief="solid",
                    borderwidth=2
                )
            else:
                # Normal appearance
                label.config(
                    bg=self.bg_color,
                    fg=self.fg_color,
                    relief="solid",
                    borderwidth=1
                )
    
    def animate_loading(self):
        """Animate the loading dots"""
        if self.is_loading:
            dots = "." * self.loading_dots
            model_name = MODELS[self.current_model_index]["name"]
            self.status_label.config(text=f"loading {model_name} {dots}")
            self.loading_dots = (self.loading_dots % 4) + 1
            self.root.after(500, self.animate_loading)
    
    def load_model(self):
        try:
            model_config = MODELS[self.current_model_index]
            logger.info(f"Loading model: {model_config['name']}")
            
            self.model = WhisperModel(
                model_config["model"],
                device=model_config["device"],
                compute_type=model_config["compute_type"]
            )
            
            self.is_loading = False
            self.text_queue.put(("status", "reverberation"))
            
            # Start recording if not reloading
            if not self.is_reloading:
                self.start_recording()
            self.is_reloading = False
            
        except Exception as e:
            self.is_loading = False
            self.is_reloading = False
            self.text_queue.put(("error", f"Error loading model: {str(e)}"))
            
    def start_recording(self):
        self.is_recording = True
        self.audio_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.transcribe_thread = threading.Thread(target=self.transcribe_audio, daemon=True)
        self.audio_thread.start()
        self.transcribe_thread.start()
        
    def update_buffer_indicator(self, progress, processing=False):
        """Update the buffer progress dots"""
        if processing:
            # Show all dots filled when processing
            for dot in self.buffer_dots:
                dot.config(text="●", fg=self.highlight_color)
        else:
            # Show progress normally
            for i, dot in enumerate(self.buffer_dots):
                if i < progress:
                    dot.config(text="●", fg=self.fg_color)  # Filled dot
                else:
                    dot.config(text="○", fg="#444444")  # Empty dot
        
    def record_audio(self):
        CHUNK = 1024  # Back to larger chunks for cleaner audio
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        p = pyaudio.PyAudio()
        stream = None
        
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            audio_buffer = []
            frames_per_chunk = RATE * 10  # 10 second chunks for complete thoughts
            overlap_frames = int(RATE * 0.5)  # Short overlap to avoid duplication
            
            while self.is_recording:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_buffer.append(data)
                    
                    # Update buffer progress indicator (make dots fill slightly earlier)
                    progress = min(9, (len(audio_buffer) * CHUNK * 10) // frames_per_chunk)
                    self.text_queue.put(("buffer_progress", progress))
                    
                    # Process chunks with small overlap to preserve word boundaries
                    if len(audio_buffer) >= frames_per_chunk // CHUNK:
                        # Signal processing state
                        self.text_queue.put(("buffer_processing", True))
                        
                        audio_data = b''.join(audio_buffer)
                        self.audio_queue.put(audio_data)
                        
                        # Keep small overlap (0.25s) to preserve word boundaries
                        overlap_chunks = overlap_frames // CHUNK
                        if len(audio_buffer) > overlap_chunks:
                            audio_buffer = audio_buffer[-overlap_chunks:]
                        else:
                            audio_buffer = []
                        
                except Exception as e:
                    if self.is_recording:  # Only log if we're still supposed to be recording
                        logger.error(f"Audio error: {e}")
                    break
                    
        finally:
            logger.info("Cleaning up audio stream")
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
            p.terminate()
            
    def transcribe_audio(self):
        last_segments = []  # Track recent segments to filter repetitions
        last_chunk_words = []  # Track words from overlap region
        recent_text_context = []  # Track recent text for context prompts
        
        while self.is_recording:
            try:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    
                    # Convert audio bytes to numpy array
                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Audio preprocessing for better sensitivity
                    # Normalize audio to use full dynamic range
                    if np.max(np.abs(audio_np)) > 0:
                        audio_np = audio_np / (np.max(np.abs(audio_np)) * 0.8)  # Normalize with headroom
                    
                    # Apply additional gain for quiet speech
                    audio_gain = 2.0  # Moderate boost after normalization
                    audio_np = np.clip(audio_np * audio_gain, -1.0, 1.0)
                    
                    # Check audio level after gain
                    audio_level = np.abs(audio_np).mean()
                    logger.debug(f"Audio level (after {audio_gain}x gain): {audio_level:.4f}")
                    
                    # Even more aggressive - process almost all audio
                    if audio_level < 0.0005:  # Extremely sensitive
                        logger.debug("Skipping very quiet audio")
                        continue
                    
                    # Build dynamic context prompt - emphasize literal transcription
                    base_prompt = "This is a highly technical statement. Transcribe only the exact words that are spoken. Do not add, interpret, or complete sentences. If speech is unclear or incomplete, transcribe only what is clearly audible."
                    
                    if recent_text_context:
                        # Add recent context (last 2 sentences)
                        context_text = " ".join(recent_text_context[-2:])
                        full_prompt = f"{base_prompt} Previous context: \"{context_text}\""
                    else:
                        full_prompt = base_prompt
                    
                    logger.debug(f"Using prompt: {full_prompt[:100]}...")
                    
                    # Balanced transcription settings with dynamic context
                    segments, _ = self.model.transcribe(
                        audio_np, 
                        beam_size=12,  # Maximum beams for comprehensive search
                        best_of=5,    # More candidates for comprehensive coverage
                        temperature=0.0,  # Fully deterministic for literal transcription
                        condition_on_previous_text=False,  # Disable audio context - using text context instead
                        no_speech_threshold=0.1,  # Very low - catch almost everything
                        compression_ratio_threshold=3.0,  # More lenient to avoid dropping speech
                        log_prob_threshold=-1.0,  # Balanced confidence requirement
                        word_timestamps=False,
                        suppress_tokens=[-1],  # Suppress special tokens
                        repetition_penalty=1.05,  # Light repetition penalty to avoid cutting off speech
                        initial_prompt=full_prompt
                    )
                    
                    # Combine all segments into one text block for better flow
                    full_text = ""
                    for segment in segments:
                        if segment.text.strip():
                            full_text += segment.text
                    
                    if full_text.strip():
                        logger.debug(f"Full transcription: '{full_text.strip()}'")
                        # Simple word-based overlap filtering
                        words = full_text.strip().split()
                        
                        if last_chunk_words and len(words) > 0:
                            # Find overlap by comparing first few words with last chunk's end
                            overlap_size = 0
                            for i in range(min(len(last_chunk_words), len(words))):
                                if words[i] == last_chunk_words[-(len(last_chunk_words)-i)]:
                                    overlap_size = len(last_chunk_words) - i
                                    break
                            
                            # Send only the new part
                            if overlap_size > 0 and overlap_size < len(words):
                                new_words = words[overlap_size:]
                                new_text = " " + " ".join(new_words)
                            else:
                                new_text = " " + " ".join(words)
                        else:
                            # First chunk
                            new_text = " ".join(words)
                        
                        # Clean up text and send with better repetition filtering
                        new_text = new_text.strip()
                        
                        # Check for repetitive patterns
                        words = new_text.split()
                        if len(words) > 3:
                            # Simple repetition detection - check if same word repeated >3 times
                            word_counts = {}
                            for word in words:
                                word_counts[word] = word_counts.get(word, 0) + 1
                            max_count = max(word_counts.values()) if word_counts else 0
                            
                            if max_count > 5:  # More lenient - allow some repetition
                                logger.debug(f"Filtered repetitive text: '{new_text}'")
                                new_text = ""
                        
                        if new_text and new_text not in last_segments:
                            logger.debug(f"Transcribed: '{new_text}'")
                            self.text_queue.put(("text", " " + new_text))
                            
                            # Reset buffer indicator after transcription
                            self.text_queue.put(("buffer_reset", True))
                            
                            # Track recent text (last 5 for better context tracking)
                            last_segments.append(new_text)
                            if len(last_segments) > 5:
                                last_segments.pop(0)
                            
                            # Add to context buffer (keep last 3 for prompts)
                            recent_text_context.append(new_text)
                            if len(recent_text_context) > 3:
                                recent_text_context.pop(0)
                        
                        # Store last 3 words for overlap detection
                        last_chunk_words = words[-3:] if len(words) >= 3 else words
                        segment_count = 1
                    else:
                        segment_count = 0
                    
                    if segment_count == 0:
                        logger.debug("No speech detected in this chunk")
                        
                else:
                    time.sleep(0.1)  # Standard sleep
                    
            except Exception as e:
                if self.is_recording:
                    logger.error(f"Transcription error: {e}")
                
    def update_ui(self):
        try:
            while not self.text_queue.empty():
                msg_type, content = self.text_queue.get_nowait()
                
                if msg_type == "status":
                    self.status_label.config(text=content)
                elif msg_type == "text":
                    self.text_display.insert(tk.END, content + " ")
                    self.text_display.see(tk.END)
                elif msg_type == "error":
                    self.status_label.config(text=content, fg="#ff0000")
                elif msg_type == "buffer_progress":
                    self.update_buffer_indicator(content, processing=False)
                elif msg_type == "buffer_processing":
                    self.update_buffer_indicator(9, processing=True)
                elif msg_type == "buffer_reset":
                    self.update_buffer_indicator(0, processing=False)
                    
        except queue.Empty:
            pass
            
        self.root.after(100, self.update_ui)
        
    def on_any_key(self, event):
        logger.debug(f"Key pressed: {event.keysym} (state: {event.state}, keycode: {event.keycode})")
        print(f"Key pressed: {event.keysym} (state: {event.state})")
        
    def on_escape(self, event):
        logger.info("Escape pressed!")
        print("Escape pressed!")
        self.cancel()
        return "break"
        
    def on_return(self, event):
        logger.info("Return pressed!")
        print("Return pressed!")
        self.insert_text()
        return "break"
        
    def on_shift_return(self, event):
        logger.info("Shift+Return pressed!")
        print("Shift+Return pressed!")
        self.copy_to_clipboard()
        return "break"
        
    def on_tab(self, event):
        logger.info("Tab pressed!")
        # Cycle to next model
        self.current_model_index = (self.current_model_index + 1) % len(MODELS)
        self.update_model_highlight()
        
        # Save config
        self.config["model_index"] = self.current_model_index
        save_config(self.config)
        
        # Reload model
        self.reload_model()
        return "break"
        
    def reload_model(self):
        """Reload the model with new selection"""
        logger.info(f"Reloading model to: {MODELS[self.current_model_index]['name']}")
        
        # Stop current recording
        self.is_recording = False
        time.sleep(0.5)  # Give threads time to stop
        
        # Clear current text
        self.text_display.delete("1.0", tk.END)
        
        # Set loading state
        self.is_loading = True
        self.is_reloading = True
        self.loading_dots = 1
        
        # Start loading animation again
        self.animate_loading()
        
        # Load new model in background
        threading.Thread(target=self.load_model, daemon=True).start()
        
        # Restart recording after model loads
        threading.Thread(target=self._restart_recording, daemon=True).start()
        
    def _restart_recording(self):
        """Helper to restart recording after model reload"""
        # Wait for model to load
        while self.is_loading:
            time.sleep(0.1)
        
        # Start recording again
        if not self.is_recording:
            self.start_recording()
        
    def get_text(self):
        return self.text_display.get("1.0", tk.END).strip()
        
    def cancel(self):
        logger.info("Cancelling and closing window")
        self.is_recording = False
        
        # Give threads time to finish
        logger.info("Waiting for threads to finish...")
        time.sleep(0.5)
        
        try:
            self.root.grab_release()  # Release keyboard grab
        except:
            pass
            
        try:
            self.root.quit()  # Exit mainloop first
            self.root.destroy()  # Then destroy window
        except:
            pass
            
        logger.info("Exiting application")
        sys.exit(0)
        
    def insert_text(self):
        text = self.get_text()
        if text:
            # Use xdotool to type the text
            self.root.withdraw()  # Hide window first
            time.sleep(0.1)  # Small delay
            subprocess.run(['xdotool', 'type', '--clearmodifiers', text])
        self.cancel()
        
    def copy_to_clipboard(self):
        text = self.get_text()
        if text:
            # Use xclip to copy to clipboard
            process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
            process.communicate(text.encode('utf-8'))
        self.cancel()
        
    def run(self):
        # Show window now that everything is configured
        self.root.deiconify()
        
        # Ensure window has focus when starting
        # Focus and grab are now handled by setup_focus_and_grab
        logger.info("Window mainloop starting")
        self.root.mainloop()

if __name__ == "__main__":
    try:
        logger.info("Starting application")
        app = TranscribeWindow()
        app.run()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise
