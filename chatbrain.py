#!/usr/bin/env python3
"""
Semantic Chat Cache - Complete GUI Application
Auto-installs Ollama, downloads models, provides chat interface
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import queue
import json
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
import requests
import subprocess
import sys
import os
import platform
import urllib.request
import zipfile
import shutil
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    user_message: str
    user_embedding: np.ndarray
    assistant_response: str
    timestamp: str
    session_id: str
    usage_count: int = 0
    feedback_score: float = 0.0

class OllamaInstaller:
    """Handles automatic Ollama installation"""
    
    @staticmethod
    def is_ollama_installed():
        """Check if Ollama is installed"""
        try:
            result = subprocess.run(['ollama', '--version'], 
                                 capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @staticmethod
    def is_ollama_running():
        """Check if Ollama service is running"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def install_ollama(progress_callback=None):
        """Install Ollama automatically"""
        system = platform.system().lower()
        
        if progress_callback:
            progress_callback("Detecting system...")
        
        try:
            if system == "windows":
                return OllamaInstaller._install_windows(progress_callback)
            elif system == "darwin":  # macOS
                return OllamaInstaller._install_macos(progress_callback)
            elif system == "linux":
                return OllamaInstaller._install_linux(progress_callback)
            else:
                raise Exception(f"Unsupported system: {system}")
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error: {str(e)}")
            return False
    
    @staticmethod
    def _install_windows(progress_callback=None):
        """Install Ollama on Windows"""
        if progress_callback:
            progress_callback("Downloading Ollama for Windows...")
        
        url = "https://github.com/ollama/ollama/releases/latest/download/ollama-windows-amd64.zip"
        temp_dir = Path.home() / "AppData" / "Local" / "Temp" / "ollama_install"
        temp_dir.mkdir(exist_ok=True)
        
        zip_path = temp_dir / "ollama.zip"
        
        # Download
        urllib.request.urlretrieve(url, zip_path)
        
        if progress_callback:
            progress_callback("Extracting Ollama...")
        
        # Extract
        install_dir = Path.home() / "AppData" / "Local" / "Programs" / "Ollama"
        install_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(install_dir)
        
        # Add to PATH (for current session)
        os.environ["PATH"] = str(install_dir) + os.pathsep + os.environ["PATH"]
        
        if progress_callback:
            progress_callback("Ollama installed successfully!")
        
        return True
    
    @staticmethod
    def _install_macos(progress_callback=None):
        """Install Ollama on macOS"""
        if progress_callback:
            progress_callback("Installing Ollama via curl...")
        
        try:
            subprocess.run(['curl', '-fsSL', 'https://ollama.ai/install.sh'], 
                         stdout=subprocess.PIPE, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    @staticmethod
    def _install_linux(progress_callback=None):
        """Install Ollama on Linux"""
        if progress_callback:
            progress_callback("Installing Ollama via curl...")
        
        try:
            subprocess.run(['bash', '-c', 'curl -fsSL https://ollama.ai/install.sh | sh'], 
                         check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    @staticmethod
    def start_ollama(progress_callback=None):
        """Start Ollama service"""
        if progress_callback:
            progress_callback("Starting Ollama service...")
        
        try:
            if platform.system().lower() == "windows":
                subprocess.Popen(['ollama', 'serve'], 
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                subprocess.Popen(['ollama', 'serve'])
            
            # Wait for service to start
            for i in range(30):  # Wait up to 30 seconds
                if OllamaInstaller.is_ollama_running():
                    if progress_callback:
                        progress_callback("Ollama service started!")
                    return True
                time.sleep(1)
                if progress_callback:
                    progress_callback(f"Waiting for Ollama to start... ({i+1}/30)")
            
            return False
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error starting Ollama: {str(e)}")
            return False
    
    @staticmethod
    def pull_model(model_name="llama3.2:3b", progress_callback=None):
        """Pull a model from Ollama"""
        if progress_callback:
            progress_callback(f"Downloading {model_name}...")
        
        try:
            # Use the API endpoint instead of CLI for better control
            response = requests.post('http://localhost:11434/api/pull', 
                                   json={"name": model_name},
                                   stream=True,
                                   timeout=300)
            
            if response.status_code != 200:
                if progress_callback:
                    progress_callback(f"Failed to start download: {response.status_code}")
                return False
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'status' in data and progress_callback:
                            status = data['status']
                            if 'completed' in data and 'total' in data:
                                percent = (data['completed'] / data['total']) * 100
                                progress_callback(f"{status}: {percent:.1f}%")
                            else:
                                progress_callback(status)
                        
                        # Check if complete
                        if data.get('status') == 'success' or 'successfully' in data.get('status', '').lower():
                            if progress_callback:
                                progress_callback(f"✅ {model_name} downloaded successfully!")
                            return True
                    except json.JSONDecodeError:
                        continue
            
            return True
            
        except requests.exceptions.Timeout:
            if progress_callback:
                progress_callback(f"Download timeout - trying CLI fallback...")
            # Fallback to CLI method
            return OllamaInstaller._pull_model_cli(model_name, progress_callback)
        except Exception as e:
            if progress_callback:
                progress_callback(f"Download error: {str(e)} - trying CLI fallback...")
            # Fallback to CLI method
            return OllamaInstaller._pull_model_cli(model_name, progress_callback)
    
    @staticmethod
    def _pull_model_cli(model_name, progress_callback=None):
        """Fallback CLI method for pulling models"""
        try:
            if progress_callback:
                progress_callback(f"Using CLI to download {model_name}...")
            
            process = subprocess.Popen(['ollama', 'pull', model_name], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT,
                                     text=True)
            
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output and progress_callback:
                    clean_output = output.strip()
                    if clean_output:
                        progress_callback(f"CLI: {clean_output}")
            
            return process.returncode == 0
        except Exception as e:
            if progress_callback:
                progress_callback(f"CLI download also failed: {str(e)}")
            return False

    @staticmethod
    def get_available_models():
        """Get list of available models to download"""
        common_models = [
            "llama3.2:1b",      # Fastest, smallest
            "llama3.2:3b",      # Good balance
            "gemma2:2b",        # Google's small model
            "qwen2.5:3b",       # Qwen small model
            "phi3:mini",        # Microsoft's mini model
        ]
        return common_models

    @staticmethod
    def get_installed_models():
        """Get list of installed models"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model.get('name', '') for model in models]
        except:
            pass
        return []

class SemanticChatCache:
    """Core semantic cache system (simplified for GUI)"""
    
    def __init__(self, cache_dir: str = "./chat_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.selected_model = "llama3.2:3b"  # Default model
        self.embed_model = None
        self.faiss_index = None
        self.conn = None
        
        self._init_database()
        self._init_embedding_model()
        self._load_or_create_faiss_index()
    
    def _init_database(self):
        """Initialize SQLite database"""
        self.db_path = self.cache_dir / "chat_history.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                feedback_score REAL DEFAULT 0.0
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_query TEXT NOT NULL,
                cached_query TEXT NOT NULL,
                cosine_similarity REAL NOT NULL,
                user_preferred TEXT NOT NULL,
                similarity_rating INTEGER NOT NULL,
                useful INTEGER NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        self.conn.commit()
    
    def _init_embedding_model(self):
        """Initialize embedding model"""
        try:
            self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embed_model = None
    
    def _load_or_create_faiss_index(self):
        """Load or create FAISS index"""
        if self.embedding_dim:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.embed_model:
            return self.embed_model.encode([text])[0]
        return np.zeros(384)  # Fallback
    
    def search_cache(self, user_query: str, k: int = 5) -> List[Tuple]:
        """Search for similar cached responses"""
        if not self.faiss_index or self.faiss_index.ntotal == 0:
            return []
        
        user_embedding = self.embed_text(user_query).reshape(1, -1)
        scores, indices = self.faiss_index.search(user_embedding, min(k, self.faiss_index.ntotal))
        
        cursor = self.conn.execute("""
            SELECT user_message, assistant_response, timestamp, session_id, usage_count, feedback_score 
            FROM chat_history ORDER BY id
        """)
        all_entries = cursor.fetchall()
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(all_entries):
                row = all_entries[idx]
                results.append((row, float(score)))
        
        return results
    
    def should_use_cache(self, similarity: float) -> Tuple[bool, str]:
        """Simple similarity-based caching decision"""
        if similarity > 0.95:
            return True, f"Excellent match ({similarity:.1%})"
        elif similarity > 0.85:
            return True, f"Very good match ({similarity:.1%})"
        elif similarity > 0.75:
            return True, f"Good match ({similarity:.1%})"
        else:
            return False, f"Similarity too low ({similarity:.1%})"
    
    def call_ollama(self, prompt: str, model: str = None) -> str:
        """Call Ollama API"""
        if model is None:
            model = getattr(self, 'selected_model', 'llama3.2:1b')
            
        try:
            response = requests.post("http://localhost:11434/api/generate", 
                                   json={
                                       "model": model,
                                       "prompt": prompt,
                                       "stream": False
                                   }, timeout=60)
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: Ollama returned status {response.status_code}"
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"
    
    def get_response(self, user_query: str) -> Dict:
        """Get response - cached or fresh"""
        # Search cache
        cache_results = self.search_cache(user_query)
        
        if cache_results:
            best_match, similarity = cache_results[0]
            should_cache, reason = self.should_use_cache(similarity)
            
            if should_cache:
                return {
                    "response": best_match[1],  # assistant_response
                    "source": "cache",
                    "similarity": similarity,
                    "reasoning": reason,
                    "cached_query": best_match[0]  # user_message
                }
        
        # Get fresh response
        llm_response = self.call_ollama(user_query)
        
        # Store new conversation
        self.store_conversation(user_query, llm_response)
        
        return {
            "response": llm_response,
            "source": "llm",
            "similarity": None,
            "reasoning": "No suitable cached response found",
            "cached_query": None
        }
    
    def store_conversation(self, user_message: str, assistant_response: str):
        """Store conversation in cache"""
        timestamp = datetime.now().isoformat()
        
        # Store in database
        self.conn.execute("""
            INSERT INTO chat_history (user_message, assistant_response, timestamp, session_id)
            VALUES (?, ?, ?, ?)
        """, (user_message, assistant_response, timestamp, "gui_session"))
        self.conn.commit()
        
        # Add to FAISS index
        if self.faiss_index is not None:
            user_embedding = self.embed_text(user_message).reshape(1, -1)
            self.faiss_index.add(user_embedding)
    
    def record_feedback(self, user_query: str, cached_query: str, helpful: bool):
        """Record user feedback"""
        similarity_rating = 5 if helpful else 1
        
        self.conn.execute("""
            INSERT INTO feedback_data 
            (user_query, cached_query, cosine_similarity, user_preferred, 
             similarity_rating, useful, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_query, cached_query, 0.9, "cached" if helpful else "llm",
            similarity_rating, int(helpful), datetime.now().isoformat()
        ))
        self.conn.commit()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM chat_history")
        total_conversations = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM feedback_data")
        total_feedback = cursor.fetchone()[0]
        
        cache_size = self.faiss_index.ntotal if self.faiss_index else 0
        
        return {
            "total_conversations": total_conversations,
            "total_feedback": total_feedback,
            "cache_size": cache_size,
            "embedding_model": "all-MiniLM-L6-v2" if self.embed_model else "Not loaded"
        }

class SemanticChatGUI:
    """Main GUI application"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Semantic Chat Cache")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2b2b2b')
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#2b2b2b')
        style.configure('Dark.TLabel', background='#2b2b2b', foreground='white')
        style.configure('Dark.TButton', background='#404040', foreground='white')
        
        # Initialize components
        self.cache = None
        self.message_queue = queue.Queue()
        
        # Setup GUI
        self.setup_gui()
        
        # Start initialization
        self.check_and_setup_ollama()
    
    def setup_gui(self):
        """Setup the main GUI"""
        # Create main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="🧠 Semantic Chat Cache", 
                               font=('Arial', 16, 'bold'), style='Dark.TLabel')
        title_label.pack(pady=(0, 10))
        
        # Status frame
        status_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(status_frame, text="Initializing...", 
                                     style='Dark.TLabel')
        self.status_label.pack(side=tk.LEFT)
        
        self.stats_label = ttk.Label(status_frame, text="", style='Dark.TLabel')
        self.stats_label.pack(side=tk.RIGHT)
        
        # Chat display
        chat_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD,
            width=80,
            height=25,
            bg='#1e1e1e',
            fg='white',
            insertbackground='white',
            font=('Consolas', 10)
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Input frame
        input_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        input_frame.pack(fill=tk.X)
        
        self.message_entry = tk.Entry(
            input_frame,
            bg='#404040',
            fg='white',
            insertbackground='white',
            font=('Arial', 11)
        )
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.message_entry.bind('<Return>', self.send_message)
        
        self.send_button = ttk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            style='Dark.TButton'
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Initially disable input
        self.message_entry.config(state='disabled')
        self.send_button.config(state='disabled')
    
    def check_and_setup_ollama(self):
        """Check Ollama installation and setup if needed"""
        def setup_thread():
            try:
                # Check if Ollama is installed
                if not OllamaInstaller.is_ollama_installed():
                    self.update_status("Ollama not found. Installing...")
                    if not OllamaInstaller.install_ollama(self.update_status):
                        self.update_status("❌ Failed to install Ollama")
                        return
                
                # Check if Ollama is running
                if not OllamaInstaller.is_ollama_running():
                    if not OllamaInstaller.start_ollama(self.update_status):
                        self.update_status("❌ Failed to start Ollama")
                        return
                
                # Check for models and download if needed
                self.update_status("Checking for models...")
                has_model, available_models = self.check_model_exists()
                
                if not has_model:
                    # Try multiple models in order of preference
                    models_to_try = ["llama3.2:1b", "gemma2:2b", "qwen2.5:3b", "phi3:mini"]
                    
                    model_downloaded = False
                    for model in models_to_try:
                        self.update_status(f"Trying to download {model}...")
                        if OllamaInstaller.pull_model(model, self.update_status):
                            model_downloaded = True
                            self.selected_model = model
                            break
                        else:
                            self.update_status(f"Failed to download {model}, trying next...")
                    
                    if not model_downloaded:
                        self.update_status("❌ Failed to download any model. Please install manually: ollama pull llama3.2:1b")
                        return
                else:
                    # Use the first available model
                    preferred_models = ['llama3.2:3b', 'llama3.2:1b', 'gemma2:2b', 'qwen2.5:3b']
                    self.selected_model = None
                    
                    for preferred in preferred_models:
                        if any(preferred in model for model in available_models):
                            self.selected_model = preferred
                            break
                    
                    if not self.selected_model and available_models:
                        self.selected_model = available_models[0]
                    
                    self.update_status(f"Using model: {self.selected_model}")

                # Initialize cache
                self.update_status("Initializing semantic cache...")
                self.cache = SemanticChatCache()
                
                # Enable chat
                self.root.after(0, self.enable_chat)
                
            except Exception as e:
                self.update_status(f"❌ Setup failed: {str(e)}")
        
        threading.Thread(target=setup_thread, daemon=True).start()
    
    def check_model_exists(self) -> Tuple[bool, List[str]]:
        """Check if any model exists and return available models"""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                # Check for preferred models
                preferred = ['llama3.2:3b', 'llama3.2:1b', 'gemma2:2b', 'qwen2.5:3b']
                for model in preferred:
                    if any(model in name for name in model_names):
                        return True, model_names
                
                # If we have any model, use it
                if model_names:
                    return True, model_names
                    
                return False, []
        except Exception as e:
            logger.error(f"Error checking models: {e}")
            return False, []
    
    def update_status(self, message: str):
        """Update status label (thread-safe)"""
        self.root.after(0, lambda: self.status_label.config(text=message))
    
    def enable_chat(self):
        """Enable chat interface"""
        self.message_entry.config(state='normal')
        self.send_button.config(state='normal')
        self.message_entry.focus()
        
        self.update_status("✅ Ready to chat!")
        self.update_stats()
        
        self.add_message("System", "🚀 Semantic Chat Cache is ready! Ask me anything.", "system")
        self.add_message("System", "💡 I'll remember our conversations and cache similar responses for faster replies.", "system")
    
    def update_stats(self):
        """Update statistics display"""
        if self.cache:
            stats = self.cache.get_stats()
            stats_text = f"💬 {stats['total_conversations']} conversations | 👍 {stats['total_feedback']} feedback | 🧠 {stats['cache_size']} cached"
            self.stats_label.config(text=stats_text)
    
    def send_message(self, event=None):
        """Send a message"""
        message = self.message_entry.get().strip()
        if not message or not self.cache:
            return
        
        # Clear input
        self.message_entry.delete(0, tk.END)
        
        # Add user message
        self.add_message("You", message, "user")
        
        # Disable input while processing
        self.message_entry.config(state='disabled')
        self.send_button.config(state='disabled')
        
        # Process in background
        def process_message():
            try:
                result = self.cache.get_response(message)
                self.root.after(0, lambda: self.handle_response(message, result))
            except Exception as e:
                self.root.after(0, lambda: self.add_message("System", f"Error: {str(e)}", "error"))
                self.root.after(0, self.enable_input)
        
        threading.Thread(target=process_message, daemon=True).start()
    
    def handle_response(self, user_message: str, result: Dict):
        """Handle the response from cache/LLM"""
        source_icon = "💾" if result["source"] == "cache" else "🤖"
        source_text = "Cached" if result["source"] == "cache" else "Fresh"
        
        # Add response
        response_text = f"{source_icon} {result['response']}"
        self.add_message("Assistant", response_text, "assistant")
        
        # Add metadata
        if result["source"] == "cache":
            meta_text = f"🔍 {result['reasoning']} | Original: \"{result['cached_query']}\""
            self.add_message("System", meta_text, "cache_info")
            
            # Add feedback buttons for cached responses
            self.add_feedback_buttons(user_message, result['cached_query'])
        else:
            meta_text = f"🔍 {result['reasoning']}"
            self.add_message("System", meta_text, "llm_info")
        
        # Update stats and re-enable input
        self.update_stats()
        self.enable_input()
    
    def add_feedback_buttons(self, user_query: str, cached_query: str):
        """Add feedback buttons for cached responses"""
        feedback_frame = tk.Frame(self.chat_display, bg='#1e1e1e')
        
        helpful_btn = tk.Button(
            feedback_frame,
            text="👍 Helpful",
            bg='#4a9eff',
            fg='white',
            border=0,
            command=lambda: self.record_feedback(user_query, cached_query, True, feedback_frame)
        )
        helpful_btn.pack(side=tk.LEFT, padx=5)
        
        not_helpful_btn = tk.Button(
            feedback_frame,
            text="👎 Not Helpful", 
            bg='#ff4a4a',
            fg='white',
            border=0,
            command=lambda: self.record_feedback(user_query, cached_query, False, feedback_frame)
        )
        not_helpful_btn.pack(side=tk.LEFT, padx=5)
        
        # Add frame to chat
        self.chat_display.window_create(tk.END, window=feedback_frame)
        self.chat_display.insert(tk.END, "\n\n")
        self.chat_display.see(tk.END)
    
    def record_feedback(self, user_query: str, cached_query: str, helpful: bool, feedback_frame: tk.Frame):
        """Record user feedback"""
        if self.cache:
            self.cache.record_feedback(user_query, cached_query, helpful)
            
            # Replace buttons with confirmation
            for widget in feedback_frame.winfo_children():
                widget.destroy()
            
            thanks_label = tk.Label(
                feedback_frame,
                text="✅ Thanks for the feedback!",
                bg='#1e1e1e',
                fg='#4a9eff',
                font=('Arial', 9)
            )
            thanks_label.pack()
            
            self.update_stats()
    
    def enable_input(self):
        """Re-enable input"""
        self.message_entry.config(state='normal')
        self.send_button.config(state='normal')
        self.message_entry.focus()
    
    def add_message(self, sender: str, message: str, msg_type: str):
        """Add a message to the chat display"""
        self.chat_display.config(state='normal')
        
        # Define colors
        colors = {
            "user": "#4a9eff",
            "assistant": "#4ade80", 
            "system": "#94a3b8",
            "cache_info": "#fbbf24",
            "llm_info": "#a78bfa",
            "error": "#ef4444"
        }
        
        color = colors.get(msg_type, "#ffffff")
        
        # Add timestamp for user/assistant messages
        if msg_type in ["user", "assistant"]:
            timestamp = datetime.now().strftime("%H:%M")
            self.chat_display.insert(tk.END, f"\n[{timestamp}] ", "timestamp")
        
        # Add sender and message
        self.chat_display.insert(tk.END, f"{sender}: ", "sender")
        self.chat_display.insert(tk.END, f"{message}\n", msg_type)
        
        # Configure tags
        self.chat_display.tag_config("timestamp", foreground="#94a3b8", font=('Arial', 8))
        self.chat_display.tag_config("sender", foreground=color, font=('Arial', 10, 'bold'))
        self.chat_display.tag_config(msg_type, foreground=color, font=('Arial', 10))
        
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        app = SemanticChatGUI()
        app.run()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main()