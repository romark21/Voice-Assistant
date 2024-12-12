import queue
import json
from vosk import Model, KaldiRecognizer
import sounddevice as sd


class SpeechRecognizer:
    def __init__(self, model_path):
        self.model = Model(model_path)
        self.device = sd.default.device
        self.sample_rate = int(sd.query_devices(self.device[0], 'input')['default_samplerate'])
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.q = queue.Queue()

    def audio_callback(self, indata, frames, time, status):
        """Обработчик потока аудио."""
        if status:
            print(f"Статус ошибки: {status}")
        self.q.put(bytes(indata))

    def listen(self):
        """Начинает потоковое распознавание речи."""
        with sd.RawInputStream(samplerate=self.sample_rate, blocksize=48000, device=sd.default.device[0],
                               dtype='int16', channels=1, callback=self.audio_callback):
            print("Говорите что-нибудь...")
            while True:
                data = self.q.get()
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())['text']
                    yield result
                # Частичные результаты (опционально)
                # else:
                #     partial_result = json.loads(self.recognizer.PartialResult()).get("partial", "")
                #     print(f"Частичное распознавание: {partial_result}")
