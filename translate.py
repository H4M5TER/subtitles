import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import stable_whisper
    import json

    return json, mo, stable_whisper


@app.cell
def _():
    enable_think = False

    id = ""
    return enable_think, id


@app.cell
def _(artifact, enable_think, json, mo, result, translate_segment):
    from copy import deepcopy

    result.convert_to_segment_level()

    history = []

    translated = deepcopy(result)
    for segment in translated.segments:
        response = translate_segment(segment.text)
        message = response.choices[0].message
        history.append(
            {
                "text": segment.text,
                "content": message.content,
                "reasoning_content": getattr(message, "reasoning_content", None),
            }
        )
        segment._default_text = message.content
        mo.output.append(message.content)

    # no need to save result since not more word level ts
    translated.to_srt_vtt(
        str(artifact.with_suffix("").with_suffix(".translate.srt")), word_level=False
    )

    if enable_think:
        with open(
            artifact.with_suffix(".chat.json"), mode="w", encoding="utf-8"
        ) as dialogue_file:
            json.dump(history, dialogue_file, ensure_ascii=False)
    return (history,)


@app.cell
def _(history, mo):
    mo.ui.table(history)
    return


@app.cell
def _(id, json, mo, stable_whisper):
    from pathlib import Path

    artifact = r"data\{id}.large-v2.regroup.json".format(id=id)
    artifact = Path(artifact)
    result = stable_whisper.WhisperResult(str(artifact))

    metadata = r"data\{id}.readable_info.json".format(id=id)
    with open(metadata, encoding="utf-8") as f:
        metadata = json.load(f)
        metadata = json.dumps(metadata, ensure_ascii=False)

    mo.output.append(result.segments)
    return artifact, metadata, result


@app.cell
def _(enable_think, metadata, result):
    import os
    from dotenv import load_dotenv
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam

    load_dotenv()

    llm = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    def translate_segment(text: str):
        messages: list[ChatCompletionMessageParam] = [  # type: ignore
            {"role": "user", "content": metadata},
            {"role": "user", "content": result.text},
            # ^ preload to prevent cache miss
            {
                "role": "system",
                "content": (
                    "请根据视频信息和全文转写翻译字幕，一次一段"
                    "只需要输出翻译，不要输出任何解释和多余的内容"
                    "段落语意可能不完整，不要输出段落没有包含的内容"
                    "保持简短、口语化，不翻译腔，保持字幕的风格和韵律"
                    "人名、地名、专有名词保持原样"
                ),
            },
            {
                "role": "user",
                "content": "what you really want to see is that there is,",
            },
            {
                "role": "assistant",
                "reasoning_content": "用户要求翻译字幕，我应该只输出本段落的翻译结果，不要输出任何解释和多余的内容",
                # ^ this make openai types unhappy
                "content": "你真正想要看到的是",
            },
            {"role": "user", "content": text},
        ]
        response = llm.chat.completions.create(
            model="deepseek-v4-flash",
            # model="deepseek-v4-pro",
            messages=messages,
            reasoning_effort="high" if enable_think else None,
            extra_body={"thinking": {"type": "disabled"}} if not enable_think else None,
            temperature=1.3,  # recommended for translation job https://api-docs.deepseek.com/zh-cn/quick_start/parameter_settings
            max_tokens=50,  # hard truncate
        )
        return response

    return (translate_segment,)


if __name__ == "__main__":
    app.run()
