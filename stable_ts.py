import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import typer
    from typer_utils import ReadableFilePath, LanguageKeyType
    from pathlib import Path

    import stable_whisper

    model_name = "large-v2"
    model = stable_whisper.load_faster_whisper(model_name)


@app.cell
def _():
    if mo.app_meta().mode == "edit":
        artifact = r""
        artifact = Path(artifact)
    return


@app.function
def regroup(artifact: ReadableFilePath):
    from regroup import regroup_en

    result = stable_whisper.WhisperResult(str(artifact))
    regroup_en(result)
    result.to_srt_vtt(str(artifact.with_suffix(".regroup.srt")), word_level=False)
    result.save_as_json(str(artifact.with_suffix(".regroup.json")))
    return result


@app.function
def transcribe(
    audio: ReadableFilePath,
    language: LanguageKeyType = None,
):
    from stable_whisper.alignment import refine

    # with mo.cache('transcribe'):
    # ^ BlockException: Unconventional formatting may lead to unexpected behavior. Please format your code, and/or reduce nesting.
    result = model.transcribe(str(audio), vad=True, regroup=False, language=language)
    refine(model, str(audio), result)
    suffix = build_suffix(language)
    result.save_as_json(str(audio.with_suffix(suffix)))
    return result


@app.function
def align(
    audio: ReadableFilePath,
    transcript: ReadableFilePath,
    language: LanguageKeyType = None,
):
    result = model.align(
        str(audio), str(transcript), vad=True, regroup=False, language=language
    )
    suffix = build_suffix(language, "aligned")
    result.save_as_json(str(audio.with_suffix(suffix)))
    return result


@app.function
def build_suffix(
    language: LanguageKeyType = None,
    work_type: str | None = None,
):
    suffix = [model_name]
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
