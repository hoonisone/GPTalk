from openai import OpenAI
from abc import ABC, abstractmethod
from typing import List, Union
from pathlib import Path
from overrides import overrides


class ScriptExtractor(ABC):
    @abstractmethod
    def __call__(self, audio_file_path:Union[str, Path]) -> List[List[str]]: # ["timestemp", "tracsription"]
        pass

class WisperScriptExtractor(ScriptExtractor):
    def __init__(self):
        self.client = OpenAI()
        
    @overrides
    def __call__(self, audio_file_path:Union[str, Path]) -> List[List[str]]: # ["timestemp", "tracsription"]
        with open(audio_file_path, 'rb') as audio_file:
            transcription = self.client.audio.transcriptions.create(model = "whisper-1", file = audio_file, response_format="srt")
            sentences = [sentence.split("\n")[1:] for sentence in transcription.split("\n\n")]
        return sentences