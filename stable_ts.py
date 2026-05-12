import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    from marimo._runtime.output import append

    import typer
    from stable_whisper.whisper_compatibility import LANGUAGES
    from typer_utils import ReadableFilePath, LanguageKeyType
    from typing import Optional, Annotated

    from pathlib import Path
    from functools import partial
    import re

    import stable_whisper
    from stable_whisper import WhisperResult
    from stable_whisper.alignment import refine
    import funasr
    from modelscope import snapshot_download

    from regroup import regroup_en


@app.function
def regroup(artifact: ReadableFilePath):
    result = stable_whisper.WhisperResult(str(artifact))
    regroup_en(result)
    result.to_srt_vtt(str(artifact.with_suffix(".regroup.srt")), word_level=False)
    result.save_as_json(str(artifact.with_suffix(".regroup.json")))
    return result


@app.function
def transcribe(
    audio: ReadableFilePath,
    model: str = "qwen-asr",
    language: LanguageKeyType = None,
    prompt: str | None = None,
):
    match model:
        case "large-v2":
            result = transcribe_whisper(audio, language, prompt)
        case "paraformer":
            result = transcribe_paraformer(audio, language, prompt)
        case _:
            raise ValueError(f"Unsupported model: {model}")
    suffix = build_suffix(language=language, model=model)
    new_filename = audio.with_suffix(suffix)
    result.save_as_json(str(new_filename))
    result.to_srt_vtt(str(new_filename.with_suffix(".srt")), word_level=False)
    return result


@app.function
@mo.cache
def get_whisper_pipeline():
    return stable_whisper.load_faster_whisper("large-v2")


@app.function
def transcribe_whisper(
    audio: ReadableFilePath,
    language: LanguageKeyType = None,
    initial_prompt: str | None = None,
    refine_ts: bool = True,
):
    whisper_pipeline = get_whisper_pipeline()
    result = whisper_pipeline.transcribe(
        str(audio),
        vad=True,
        regroup=False,
        language=language,
        initial_prompt=initial_prompt,
        # 注意输入无标点空格分割的单词列表作为 prompt
        # 会导致whisper输出无标点的结果
    )
    if refine_ts:
        refine(whisper_pipeline, str(audio), result)
    return result


@app.function
@mo.cache
def get_paraformer_pipeline() -> funasr.AutoModel:
    return funasr.AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc-c",
        spk_model="cam++",
    )


@app.function
def transcribe_paraformer(
    audio: ReadableFilePath,
    language: LanguageKeyType = "zh",
    hotword: str | None = None,
):
    pipeline = get_paraformer_pipeline()
    results = pipeline.generate(
        input=str(audio),
        language=language,
        hotword=hotword,
        return_raw_text=True,  # only available with punc_model
    )
    result = results[0]

    full_words = re.split(" ", result["raw_text"])
    word_count = 0
    segments = []
    for sentence in result["sentence_info"]:
        timestamp = sentence["timestamp"]
        n_words = len(timestamp)
        words = []
        for i, word in enumerate(full_words[word_count : word_count + n_words]):
            words.append(
                dict(
                    word=word,
                    start=timestamp[i][0] / 1000.0,
                    end=timestamp[i][1] / 1000.0,
                )
            )
        word_count += n_words

        segments.append(
            dict(
                text=sentence["text"],
                start=sentence["start"] / 1000.0,
                end=sentence["end"] / 1000.0,
                words=words,
            )
        )

    return stable_whisper.transcribe_any(
        lambda audio: {
            "language": language,
            "text": result["text"],
            "segments": segments,
        },
        str(audio),
    )


@app.function
@mo.cache
def get_qwen_asr_pipeline() -> funasr.AutoModel:
    forced_aligner = snapshot_download("Qwen/Qwen3-ForcedAligner-0.6B")

    model = funasr.AutoModel(
        model="Qwen/Qwen3-ASR-1.7B",
        forced_aligner=forced_aligner,  # this dont respect hub settings, handled by qwen-asr not by funasr
        vad_model="fsmn-vad",
        spk_model="cam++",
        # punc_model="ct-punc-c",
        # 自带标点，不需要 punc_model
        # 不论英语还是普通话，自带的标点够用，但断句效果都很差
        # 识别英语时，启用 punc_model 会给句子末尾加上全角逗号或者句号，几乎没有影响
        # 识别普通话时，启用 punc_model 会崩溃
        # funasr\auto\auto_model.py:883 in inference_with_vad
        # funasr\models\campplus\utils.py:270 in distribute_spk
        # overlap = max(min(sentence_end, spk_ed) - max(sentence_start, spk_st), 0)
        # TypeError: '>' not supported between instances of 'float' and 'NoneType'
        # 我们需要一个分词过后的列表来映射到模型给出的时间戳
        # 使用 paraformer 启用 punc_model 时，raw_text 是按空格分割的分词结果
        # 没有分词结果就必须自己重建
        # 在找到好的做法之前，我不愿意单独为 qwen-asr 这么做
        device="cuda:0",
        dtype="bf16",
    )
    return model


@app.function
def align(
    audio: ReadableFilePath,
    transcript: ReadableFilePath,
    language: LanguageKeyType = None,
):
    whisper_pipeline = get_whisper_pipeline()
    result = whisper_pipeline.align(
        str(audio), str(transcript), vad=True, regroup=False, language=language
    )
    suffix = build_suffix(language=language, work_type="aligned")
    result.save_as_json(str(audio.with_suffix(suffix)))
    return result


@app.function
def build_suffix(
    language: LanguageKeyType = None,
    model: str | None = None,
    work_type: str | None = None,
):
    suffix = []
    if model:
        suffix.append(model)
    if language:
        suffix.append(language)
    if work_type:
        suffix.append(work_type)
    suffix.append("json")
    suffix = [f".{s}" for s in suffix]
    return "".join(suffix)


@app.cell
def _():
    if mo.app_meta().mode == "script":
        _typer = typer.Typer()
        _typer.command()(transcribe)
        _typer.command()(align)
        _typer.command()(regroup)
        _typer()
    return


if __name__ == "__main__":
    app.run()
