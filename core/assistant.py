from commands import basic
from core.text_to_speech import TextToSpeech


class VoiceAssistant:
    def __init__(self, recognizer, vectorizer, clf):
        self.recognizer = recognizer
        self.vectorizer = vectorizer
        self.clf = clf
        self.tts = TextToSpeech(language='ru')

    def recognize_command(self, text):
        """Обрабатывает текст и выполняет команду."""
        try:
            text_vector = self.vectorizer.transform([text]).toarray()[0]
            return self.clf.predict([text_vector])[0]
        except Exception as e:
            print(f"Ошибка распознавания команды: {e}")
            return None

    def run(self):
        """Запускает голосового помощника."""
        print("Голосовой помощник запущен.")
        for result in self.recognizer.listen():
            try:
                if not result:
                    continue
                print(f"Распознано: {result}")
                words = result.lower().split()
                trigger_word = basic.TRIGGER.intersection(words)
                if trigger_word:
                    # Если ключевое слово найдено, отвечаем
                    trigger_word = list(trigger_word)[0]  # Получаем конкретное слово
                    response = basic.data_set.get(trigger_word)
                    self.tts.speak(response)

                    # Убираем ключевое слово из текста
                    text_without_trigger = result.lower().replace(trigger_word, '').strip()

                    command = self.recognize_command(text_without_trigger)
                    if command:
                        print(f"Команда: {command}")
                        self.tts.speak("Выполняю команду!")
                    else:
                        print("Команда не распознана.")
                else:
                    # Если ключевое слово не найдено
                    print("Имя не распознано.")
            except Exception as e:
                print(f"Ошибка во время обработки: {e}")
