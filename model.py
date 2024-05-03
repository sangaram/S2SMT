import torch
from torch import nn
from transformers import AutoProcessor, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from speechbrain.inference.classifiers import EncoderClassifier
from typing import List
from TTS.api import TTS
from pathlib import Path
import soundfile as sf
import os

class LanguageIdentifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.identifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")

    @torch.no_grad()
    def forward(self, speech: torch.Tensor):
        language_id =  self.identifier.classify_batch(speech)[-1][0].split(':')[0]
        return language_id


class ASRModel(nn.Module):
    def __init__(self, lang='en', sampling_rate=16_000):
        super(ASRModel, self).__init__()
        assert lang in ['en', 'fr']

        if lang == 'en':
            self.speech_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        else:
            self.speech_processor = Wav2Vec2Processor.from_pretrained("bhuang/asr-wav2vec2-french")
            self.model = Wav2Vec2ForCTC.from_pretrained("bhuang/asr-wav2vec2-french")
        
        self.sampling_rate = sampling_rate
        self.lang = lang
    
    def forward(self, speech:torch.Tensor):
        logits = self.model(speech).logits
        return logits

    @torch.no_grad()
    def transcribe(self, speech:torch.Tensor):
        logits = self(speech)
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = list(map(lambda x: x.lower(), self.speech_processor.batch_decode(predicted_ids)))
        return transcription
    
class TranslationModel:
    def __init__(self, device="cpu", src='en', trg='fr'):
        assert src in ['en', 'fr']
        assert trg in ['en', 'fr']
        assert src != trg
        self.src = src
        self.trg = trg
        self.tokenizer = AutoTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-{src}-{trg}")
        self.translator = AutoModelForSeq2SeqLM.from_pretrained(f"Helsinki-NLP/opus-mt-{src}-{trg}").to(device)
        self.device = device
    
    @torch.no_grad()
    def translate(self, text_batch):
        batch = self.tokenizer(text_batch, padding=True, return_tensors="pt").to(self.device)
        generated_ids = self.translator.generate(**batch)
        translation = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return translation
    
    def __call__(self, text_batch):
        return self.translate(text_batch)
    
class TTSModel:
    def __init__(self, lang='en', device="cpu"):
        self.lang = lang
        self.synthesizer = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.device = device
        
    def synthesize_speech(self, text:str, speaker_path:str):
        speech = self.synthesizer.tts(text=text, speaker_wav=speaker_path, language=self.lang)

        os.unlink(speaker_path) # Removing the temporary source speaker audio file
            
        return speech


    def batch_synthesize_speech(self, text_batch:List[str], speaker_paths:List[str]):
        return list(map(self.synthesize_speech, text_batch, speaker_paths))

class Speech2SpeechTranslationModel:
    def __init__(self, src='en', trg='fr', sampling_rate=16_000, device="cpu"):
        self.src = src
        self.trg = trg
        self.asr_model = ASRModel(lang=src, sampling_rate=sampling_rate).to(device)
        self.translation_model = TranslationModel(src=src, trg=trg, device=device)
        self.tts_model = TTSModel(lang=trg, device=device)
        self.speaker_dir = Path(".temp/speakers")
        if not self.speaker_dir.exists():
            self.speaker_dir.mkdir(parents=True)

    def s2tt(self, speech:torch.Tensor):
        text_batch = self.asr_model.transcribe(speech)
        translations = self.translation_model(text_batch)
        count = len(translations)
        speaker_paths = [self.speaker_dir / f"temp_speaker{i}.wav" for i in range(count)]
        for speech, path in zip(speech, speaker_paths):
            sf.write(path, speech.cpu().numpy(), self.asr_model.sampling_rate)

        return translations, speaker_paths

    def tts(self, translations, speaker_paths=None):
        if speaker_paths is None:
            speaker_paths = [None] * len(translations)
        elif len(speaker_paths) == 1:
            speaker_paths = speaker_paths * len(translations)
        
        targets = self.tts_model.batch_synthesize_speech(text_batch=translations, speaker_paths=speaker_paths)
        return targets

    def translate(self, speech:torch.Tensor):
        translations, speaker_paths = self.s2tt(speech)
        targets = self.tts(translations, speaker_paths)
        return targets