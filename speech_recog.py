import speech_recognition as sr
import pyttsx3 as pt
import pywhatkit as pk

listener = sr.Recognizer()
engine = pt.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

def hear():
    cmd = ""  # default value
    try:
        with sr.Microphone() as mic:
            print("Listening...")
            listener.adjust_for_ambient_noise(mic, duration=1)  # helps with background noise
            voice = listener.listen(mic, timeout=5, phrase_time_limit=5)
            print("Audio captured, recognizing...")
            cmd = listener.recognize_google(voice)
            print("Raw recognition:", cmd)  # DEBUG print
            cmd = cmd.lower()
            if 'max' in cmd:
                cmd = cmd.replace('max', '')
    except sr.WaitTimeoutError:
        print("⏳ No speech detected within timeout")
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
    except sr.RequestError as e:
        print(f"⚠️ API unavailable: {e}")
    except Exception as e:
        print(f"Other error: {e}")
    return cmd


def run():
    cmd = hear()
    print("Command received:", cmd)
    if 'play' in cmd:
        song = cmd.replace('play', '')
        speak('playing ' + song)
        pk.playonyt(song)  # song only, not "Playing song"

run()
