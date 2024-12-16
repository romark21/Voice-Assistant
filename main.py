from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from core.speech_recogn import SpeechRecognizer
from core.assistant import VoiceAssistant
from commands import basic
import os

# Загрузка переменных окружения
from dotenv import load_dotenv
load_dotenv()


def main():
    # Параметры
    vosk_model_path = "models/vosk_model"

    # Создаем распознаватель речи
    recognizer = SpeechRecognizer(vosk_model_path)

    # Настраиваем обработку команд
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(list(basic.data_set.keys()))

    clf = LogisticRegression()
    clf.fit(vectors, list(basic.data_set.values()))

    # Создаем голосового помощника
    assistant = VoiceAssistant(recognizer, vectorizer, clf)

    # Запускаем помощника
    try:
        assistant.run()
    except KeyboardInterrupt:
        print("Программа завершена пользователем.")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
