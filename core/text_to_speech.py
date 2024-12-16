import os
import torch
import sounddevice as sd
from gtts import gTTS
from io import BytesIO
import soundfile as sf
import logging
import numpy as np


class TextToSpeech:
    def __init__(self, language="ru", sample_rate=48000):
        self.language = language
        self.sample_rate = sample_rate
        self.model_dir = "D:\Phyton_programs\VoiceAssistant\models\silero_models"
        self.device = torch.device('cpu')
        self.speaker = 'kseniya'  # Default speaker for Russian

        # Silero model URLs
        self.models_urls = {
            "ru": "https://models.silero.ai/models/tts/ru/v3_1_ru.pt",
            "en": "https://models.silero.ai/models/tts/en/v3_en.pt"
        }

        # Initialize model storage path
        self.model_path = os.path.join(self.model_dir, self.language, 'model.pt')

        # Load model
        self.model = self._load_model()

    def _load_model(self):
        """Load or download the Silero model."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Download model if it does not exist
        if not os.path.isfile(self.model_path):
            logging.info(f"Downloading {self.language} model...")
            torch.hub.download_url_to_file(self.models_urls[self.language], self.model_path)

        # Load model
        try:
            model = torch.package.PackageImporter(self.model_path).load_pickle("tts_models", "model")
            model.to(self.device)
            logging.info(f"{self.language.capitalize()} model loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Error loading Silero model: {e}")
            return None

    def speak_silero(self, text):
        """Generate speech using Silero."""
        if self.model is None:
            logging.error("Silero model is not available.")
            return

        try:
            audio = self.model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=self.sample_rate
            )
            silence = np.zeros(int(0.5 * self.sample_rate))
            audio_with_silence = np.concatenate((audio, silence))
            sd.play(audio_with_silence, samplerate=self.sample_rate, blocking=True)
        except Exception as e:
            logging.error(f"Error during Silero synthesis: {e}")

    def speak_gtts(self, text):
        """Fallback to Google TTS."""
        try:
            lang = self.language if self.language in ['ru', 'en'] else 'en'
            with BytesIO() as f:
                gTTS(text=text, lang=lang, slow=False).write_to_fp(f)
                f.seek(0)
                data, fs = sf.read(f)
                sd.play(data, fs, blocking=True)
        except Exception as e:
            logging.error(f"Error during gTTS synthesis: {e}")

    def speak(self, text):
        """Main method to generate speech with fallback."""
        if self.model:
            self.speak_silero(text)
        else:
            logging.warning("Falling back to Google TTS...")
            self.speak_gtts(text)


# a = TextToSpeech()
# a.speak('Привет, Вася как дела?')
