import speech_recognition as sr

r = sr.Recognizer()

# Use default microphone
mic = sr.Microphone()

with mic as source:
    print("Adjusting for noise...")
    r.adjust_for_ambient_noise(source, duration=1)

    print("Say something!")

    try:
        audio = r.listen(
            source,
            timeout=10,
            phrase_time_limit=5
        )

        text = r.recognize_google(audio)
        print("You said:", text)

    except sr.WaitTimeoutError:
        print("No speech detected.")

    except sr.UnknownValueError:
        print("Could not understand audio.")

    except sr.RequestError as e:
        print("Google API error:", e)