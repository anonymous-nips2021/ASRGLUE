import io
import os
import sys
def transcribe_file(speech_file):
    """Transcribe the given audio file asynchronously."""
    from google.cloud import speech
    client = speech.SpeechClient()
    idx = speech_file.split()[0]
    speech_file = speech_file.split()[1]
    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    """
     Note that transcription is limited to a 60 seconds audio file.
     Use a GCS file for audio longer than 1 minute.
    """
    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        model="video"
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    response = operation.result(timeout=90)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        print("{} {}".format(idx, result.alternatives[0].transcript))

scp_path = sys.argv[1].strip()
scp = open(scp_path)
for item in scp:
  item = item.strip()
  transcribe_file(item)
