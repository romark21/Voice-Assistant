from commands import basic


class VoiceAssistant:
    def __init__(self, recognizer, vectorizer, clf):
        self.recognizer = recognizer
        self.vectorizer = vectorizer
        self.clf = clf

    def recognize_command(self, text):
        """Обрабатывает текст и выполняет команду."""
        trg = basic.TRIGGER.intersection(text.split())
        if not trg:
            return None

        text = text.replace(list(trg)[0], '').strip()
        text_vector = self.vectorizer.transform([text]).toarray()[0]
        return self.clf.predict([text_vector])[0]

    def run(self):
        """Запускает голосового помощника."""
        print("Голосовой помощник запущен.")
        for result in self.recognizer.listen():
            print(f"Распознано: {result}")
            command = self.recognize_command(result)
            if command:
                print(f"Команда: {command}")
            else:
                print("Команда не распознана.")
