import queue
from vosk import Model, KaldiRecognizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import sounddevice as sd
from commands import basic
import json

q = queue.Queue()
device = sd.default.device
sample_rate = int(sd.query_devices(device[0], 'input')['default_samplerate'])


def callback(indata, frames, time, status):
    if status:
        print(f'Статус ошибки: {status}')
    q.put(bytes(indata))


def recognize(data, vectorizer, clf):
    trg = basic.TRIGGER.intersection(data.split())
    if not trg:
        return

    data.replace(list(trg)[0], '')
    text_vector = vectorizer.transform([data]).toarray()[0]
    answer = clf.predict([text_vector])[0]
    print(answer)


def main():
    model = Model("vosk_model")
    recognizer = KaldiRecognizer(model, sample_rate)
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(list(basic.data_set.keys()))

    clf = LogisticRegression()
    clf.fit(vectors, list(basic.data_set.values()))
    del basic.data_set

    print("Запуск распознавания. Говорите что-нибудь...")

    try:
        with sd.RawInputStream(samplerate=sample_rate, blocksize=48000, device=device[0],
                               dtype='int16', channels=1, callback=callback):
            while True:
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())['text']
                    print(f"Распознано: {result}")
                    recognize(result, vectorizer, clf)
                # else:
                #     partial_result = json.loads(recognizer.PartialResult())
                #     print(f"Частичное распознавание: {partial_result.get('partial', '')}", end="\r")

    except KeyboardInterrupt:
        print("Программа завершена пользователем.")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()


