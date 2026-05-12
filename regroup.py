import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import stable_whisper
    import math
    import numpy as np


@app.function
def regroup_en(result: stable_whisper.WhisperResult):
    result.reset()
    result.ignore_special_periods()
    result.merge_all_segments()
    result.split_by_gap(1.0)
    result.split_by_punctuation(
        [(".", " "), "。", "?", "？"],  # type: ignore
        lock=True,
    )
    result.custom_operation("len=text", ">", 50, split_comma, word_level=False)
    result.custom_operation("len=text", ">", 50, split_clause, word_level=False)
    result.custom_operation("len=text", ">", 50, split_fanboy, word_level=False)

    exceeded = [s for s in result.segments if len(s.text) > 80]
    mo.output.append(exceeded)
    result.split_by_length(80)
    result.clamp_max()

    return result


@app.function
def recursive_split_segment(
    result: stable_whisper.WhisperResult,
    seg_index,
    candidates: list[int],
    max_chars=60,
):
    segment = result[seg_index]
    if not segment.has_words:
        return
    words = segment.words

    split_indices: list[int] = []

    def split(start: int, end: int, candidates: list[int]):
        candidates = [i for i in candidates if start <= i < end - 1]
        if len(candidates) == 0:
            return
        cumsum = np.cumsum([len(words[i].word) for i in range(start, end)])  # type: ignore
        char_total = cumsum[-1]
        if char_total <= max_chars:
            return
        k = math.ceil(char_total / max_chars)
        target = char_total / k

        def distance(i: int) -> int:
            mod = cumsum[i - start] % target
            return min(mod, target - mod)

        sorted_candidates: list[int] = sorted(candidates, key=distance)
        filtered_candidates = [
            i
            for i in sorted_candidates
            if min(cumsum[i - start], char_total - cumsum[i - start]) >= 20
        ]
        if not filtered_candidates:
            return
        split_index = filtered_candidates[0]
        split_indices.append(split_index)

        split(start, split_index + 1, candidates)
        split(split_index + 1, end, candidates)

    split(0, len(words), candidates)  # type: ignore
    result.split_segment_by_index(segment, sorted(split_indices), reassign_ids=False)


@app.function
def split_comma(result: stable_whisper.WhisperResult, seg_index, word_index):
    candidates = result[seg_index].get_punctuation_indices([(",", " "), "，"])
    # get_punctuation_indices excluded locked indices
    recursive_split_segment(result, seg_index, candidates)


@app.function
def split_conjunctions(result: stable_whisper.WhisperResult, seg_index, conjunctions):
    candidates = []
    segment = result[seg_index]
    locked_split_indices = segment.get_locked_indices()
    for conj in conjunctions:
        for split_index, word_timing in enumerate(segment.words[1:]):
            # split_index equals real word index - 1
            # because we want split before conj
            if split_index in locked_split_indices:
                continue
            if word_timing.word.strip() == conj:
                # just strip spaces
                # ignore words with punctuation
                # case sensitive
                candidates.append(split_index)
    recursive_split_segment(result, seg_index, candidates)


@app.function
def split_clause(result: stable_whisper.WhisperResult, seg_index, word_index):
    clause_conjunctions = [
        "that",
        "which",
        "what",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        # "why",
        "because",
        "since",
        "while",
        "if",
        "although",
        "though",
        # "as",
    ]
    split_conjunctions(result, seg_index, clause_conjunctions)


@app.function
def split_fanboy(result: stable_whisper.WhisperResult, seg_index, word_index):
    coordinating_conjunctions = [
        "for",
        "and",
        "nor",
        "but",
        "or",
        "yet",
        # "so",
    ]
    split_conjunctions(result, seg_index, coordinating_conjunctions)


@app.function
def split_preposition(result: stable_whisper.WhisperResult, seg_index, word_index):
    prepositions = [
        "in",
        "on",
        "at",
        "about",
        "with",
        "without",
        "to",
        "of",
        "by",
        "from",
    ]
    split_conjunctions(result, seg_index, prepositions)


if __name__ == "__main__":
    app.run()
