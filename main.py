import torch
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoProcessor, Wav2Vec2Processor, Wav2Vec2ForCTC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import evaluate
from sacrebleu.metrics import BLEU
from model import Speech2SpeechTranslationModel
from datasets import load_dataset, Audio
import soundfile as sf
from pathlib import Path
import tqdm
from argparse import ArgumentParser
import random
import os
import warnings
warnings.filterwarnings("ignore")

### Parsing CLI arguments
parser = ArgumentParser()
parser.add_argument('--mode', type=str, choices=['infer', 'random_sampling', 'eval'], default='infer')
parser.add_argument('--audio', type=str, default=None, help='Input audio on which to run inference.')
parser.add_argument('--src_lang', type=str, default='en')
parser.add_argument('--trg_lang', type=str, default='fr')
parser.add_argument('--subset', type=str, default='clean', help='Dataset subset to use. Used only for English Librispeech dataset. Ignored for the others')
parser.add_argument('--n_samples', type=int, default=5)
parser.add_argument('--patience', type=int, default=5)
args = parser.parse_args()
###


### Model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Speech2SpeechTranslationModel(src=args.src_lang, trg=args.trg_lang, device=device)
###

### Utilities
wer = evaluate.load("wer")
#sacrebleu = evaluate.load("sacrebleu")

def uppercase(example):
    return {"transcription": example["transcription"].upper()}

def prepare_dataset(batch):
    audio = batch["audio"]
    batch = model.asr_model.speech_processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
    batch["input_length"] = len(batch["input_values"][0])
    return batch

def prepare_dataset_for_s2st(batch):
    audio = batch["audio"]
    text = batch["text"]
    batch = model.asr_model.speech_processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["text"])
    batch["input_length"] = len(batch["input_values"][0])
    synthetic_translation = model.translation_model([text.lower()])[0]
    batch["text_translation"] = synthetic_translation.lower()
    return batch

def compute_metrics(logits, label_ids):
    pred_logits = logits.cpu().numpy()
    pred_ids = np.argmax(pred_logits, axis=-1)

    label_ids[label_ids == -100] = model.asr_model.speech_processor.tokenizer.pad_token_id

    pred_str = model.asr_model.speech_processor.batch_decode(pred_ids)
    label_str = model.asr_model.speech_processor.batch_decode(label_ids, group_tokens=False)

    wer_value = wer.compute(predictions=pred_str, references=label_str)

    return wer_value

@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

@dataclass
class DataCollatorBLEUWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        batch["translations"] = [feature["text_translation"] for feature in features]

        return batch
###

if args.mode == 'infer':
    audio, sample_rate = sf.read(args.audio, dtype='float32')
    audio = torchaudio.functional.resample(audio, sample_rate, 16_000)
    input_values = model.asr_model.speech_processor(audio, return_tensors="pt", padding="longest").input_values.to(device)
    audio_translation =  model.translate(input_values)
    sf.write('out.wav', np.array(audio_translation[0]), samplerate=24_000)
elif args.mode == 'random_sampling':
    ### Dataset loading
    #dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
    if args.src_lang == 'en':
        dataset = load_dataset("librispeech_asr", args.subset, split="test")
    elif args.src_lang == 'fr':
        dataset = load_dataset("facebook/multilingual_librispeech", "french", split="test")
    else:
        raise NotImplementedError(f'Language "{args.src_lang}" not implemented.')
    
    N = len(dataset)
    sample_ids = random.sample(list(range(N)), k=args.n_samples)
    samples = dataset.select(sample_ids)
    ###

    ### Building audio batch
    audio_batch = model.asr_model.speech_processor.pad(
        [{"input_values": sample["audio"]["array"]} for sample in samples],
        padding="longest",
        return_tensors="pt"
    ).input_values.to(device)
    ###

    ### Output destination folder
    dest_folder = Path("./output")
    if not dest_folder.exists():
        dest_folder.mkdir(parents=True)
    ###

    ### Inference
    audio_translations = model.translate(audio_batch)
    for i, audio in enumerate(audio_translations):
        audio_file = dest_folder / f"output{i}.wav"
        sf.write(audio_file, np.array(audio), samplerate=24_000)
    ###
elif args.mode == 'eval':
    if args.src_lang == 'en':
        dataset = load_dataset('librispeech_asr', args.subset, split='test')
        target_processor = Wav2Vec2Processor.from_pretrained("bhuang/asr-wav2vec2-french")
        target_model = Wav2Vec2ForCTC.from_pretrained("bhuang/asr-wav2vec2-french").to(device)
    elif args.src_lang == 'fr':
        dataset = load_dataset("facebook/multilingual_librispeech", "french", split="test")
        target_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        target_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    else:
        raise NotImplementedError(f'Language "{args.src_lang}" not implemented.')
    encoded_data = dataset.map(prepare_dataset_for_s2st)
    data_collator = DataCollatorBLEUWithPadding(processor=model.asr_model.speech_processor, padding="longest")
    data_loader = DataLoader(
        encoded_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    resampler = torchaudio.transforms.Resample(24000, 16000)

    refs = []
    preds = []
    post_asr_preds = []
    wer_avg = 0.
    with torch.no_grad():
        for batch in tqdm.auto.tqdm(data_loader):
            batch_text_translations, speaker_paths = model.s2tt(batch.input_values.to(device))
            preds.extend(batch_text_translations)
            batch_translations = model.tts(batch_text_translations, speaker_paths)
            batch_translations = target_processor.pad([
                {"input_values": resampler(torch.tensor(audio))} for audio in batch_translations
            ])

            logits = target_model(batch_translations.input_values.to(device)).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            texts = target_processor.batch_decode(predicted_ids)

            refs.extend(batch.translations)
            post_asr_preds.extend(texts)

            logits = model.asr_model(batch.input_values.to(device))
            wer_avg += compute_metrics(logits, batch.labels)

    bleu = BLEU(lowercase=True)
    print(f"BLEU: {bleu.corpus_score(preds, refs)}")
    print(f"ASR-BLEU: {bleu.corpus_score(post_asr_preds, refs)}")
    wer_avg /= len(data_loader)
    print(f"WER: {wer_avg}")

else:
    raise NotImplemented(f'Mode {args.mode} not implemented')
