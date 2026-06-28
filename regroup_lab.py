import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    from marimo._runtime.output import append

    import re
    from pathlib import Path

    import math
    import numpy as np
    import polars as pl

    import plotly.express as px

    import stable_whisper
    from stable_whisper import WhisperResult
    import spacy
    import benepar
    import nltk
    from nltk import Tree, ParentedTree
    from nltk.tree.prettyprinter import TreePrettyPrinter


@app.function
@mo.cache
def get_benepar_pipeline() -> spacy.Language:
    nlp = spacy.load("en_core_web_md")
    nlp.add_pipe("benepar", config={"model": "benepar_en3_large"})
    return nlp


@app.cell(column=1, hide_code=True)
def _():
    widget_len_target = mo.ui.number(1, 100, value=50, step=1, label="Target chars")
    widget_len_floor = mo.ui.number(1, 100, value=10, step=1, label="Floor chars")

    widget_semantic = mo.ui.number(
        0, 1000, value=100, step=1, label="Semantic weight (%)"
    )
    widget_split_penalty = mo.ui.number(
        0, 1000, value=100, step=1, label="Split penalty weight (%)"
    )

    mo.vstack(
        [
            mo.hstack([widget_len_target, widget_len_floor]),
            mo.hstack([widget_semantic, widget_split_penalty]),
        ]
    )
    return (
        widget_len_floor,
        widget_len_target,
        widget_semantic,
        widget_split_penalty,
    )


@app.cell
def _(result, segment_select, sents, use_input):
    if not use_input.value:
        sent_id = segment_select.value
        sent = sents[sent_id]
        n_tokens = len(sent)
        n_bounds = n_tokens - 1
        segment = result[sent_id]
    return n_tokens, segment, sent


@app.cell
def _(sent, text_input, use_input):
    if use_input.value:
        df = prepare(text_input.value)
    else:
        df = prepare(sent)
    df
    return (df,)


@app.function
def prepare_batch(result_path: Path) -> tuple[WhisperResult, list["Span"]]:
    result = WhisperResult(str(result_path))
    result.reset()
    result.merge_all_segments()

    nlp = get_benepar_pipeline()
    doc = nlp(result.text)
    sents = list(doc.sents)
    indices: list[int] = []
    for s in set(s.text for s in sents):
        matches = result.find(re.escape(s))
        for match in matches:
            word_indices = match.word_indices
            indices.append(word_indices[0][-1])
    indices = list(set(indices))
    indices = sorted(indices)
    result.split_segment_by_index(result[0], indices)
    result.clamp_max()
    return result, sents


@app.cell
def _(result):
    [s.text for s in result.segments]
    return


@app.cell(column=2)
def _():
    result_path = mo.ui.text("")
    use_input = mo.ui.switch(value=False, label="Use input")
    mo.vstack([result_path, use_input])
    return result_path, use_input


@app.cell
def _(result_path, use_input):
    result = []
    if not use_input.value:
        result, sents = prepare_batch(Path(result_path.value))
    segment_select = mo.ui.number(0, len(result), step=1, label="Segment index")
    text_input = mo.ui.text_area(
        value="",
        label="输入句子",
        full_width=True,
    )

    mo.vstack([segment_select, text_input])
    return result, segment_select, sents, text_input


@app.cell
def _(result, segment, segment_select, sent, spacy_bounds):
    whisper_bounds = spacy_to_whisper(spacy_bounds, sent, segment)
    whisper_group = (
        pl.DataFrame([wt.word for wt in result[segment_select.value].words])
        .with_row_index()
        .with_columns(pl.col("index").cut(breaks=whisper_bounds).alias("group"))
        .group_by("group", maintain_order=True)
        .agg(pl.col("column_0").str.join(" ").alias("segment"))
    )
    whisper_group.select("segment").to_series().to_list()
    # whisper_group
    # whisper_bounds
    return


@app.cell
def _(df, spacy_bounds):
    spacy_group = (
        df.select("spacy_tokens")
        .with_row_index()
        .with_columns(pl.col("index").cut(breaks=spacy_bounds).alias("group"))
        .group_by("group", maintain_order=True)
        .agg(pl.col("spacy_tokens").str.join(" ").alias("segment"))
    )
    spacy_group.select("segment").to_series().to_list()
    # spacy_group
    # spacy_bounds
    return


@app.function
def spacy_to_whisper(
    spacy_bounds: list[int], sent: list["Span"], segment: stable_whisper.Segment
) -> list[int]:
    char_len = [len(wordtiming.word) for wordtiming in segment.words]
    cumsum = np.cumsum(char_len)
    whisper_bounds = [
        sent[i].idx - sent.start_char
        #       ^ doc level
        for i in spacy_bounds
    ]
    whisper_bounds = [np.searchsorted(cumsum, i, side="right") for i in whisper_bounds]
    whisper_bounds = [i for i in whisper_bounds if 0 <= i < len(segment.words)]
    whisper_bounds = sorted(list(set(whisper_bounds)))
    # unsort indices break splitting by stable_whisper
    return whisper_bounds


@app.cell(column=3)
def _(
    widget_len_floor,
    widget_len_target,
    widget_semantic,
    widget_split_penalty,
):
    len_target = widget_len_target.value
    len_floor = widget_len_floor.value

    semantic_weight = widget_semantic.value / 100
    split_penalty_weight = widget_split_penalty.value / 100
    return (
        len_floor,
        len_target,
        semantic_weight,
        split_penalty_weight,
    )


@app.cell
def _(
    df,
    len_floor,
    len_target,
    n_tokens,
    semantic_weight,
    split_penalty_weight,
):
    spacy_bounds, transition, dp, prev = comp_dp(
        df, len_target, len_floor, semantic_weight, split_penalty_weight
    )
    transition = pl.from_numpy(
        transition,
        schema=df.with_row_index()
        .with_columns(
            [
                pl.concat_str(
                    pl.col("index"),
                    pl.col("spacy_tokens"),
                ).alias("word")
            ]
        )
        .select("word")
        .to_series()
        .to_list(),
    )
    px.imshow(transition)
    # transition
    return spacy_bounds, transition


@app.cell
def _(df):
    bound_before_labels = [
        f'{i},"{token}",' for i, token in enumerate(df["spacy_tokens"].to_list())
    ]
    return (bound_before_labels,)


@app.cell
def _(bound_before_labels, df):
    px.line(
        df,
        x=bound_before_labels,
        # y=[c for c in df.columns if c != "x"],
        y=[
            prefix + c
            for c in (
                "dist_tree",
                "dist_depth",
                "semantic_cost",
            )
            for prefix in (
                "",
                "z_",
            )
        ],
    )
    return


@app.cell(column=4)
def _(len_floor, len_target):
    from functools import partial

    len_cost_col = (
        pl.col("n_char")
        .map_elements(partial(len_cost, target=len_target, floor=len_floor))
        .alias("len_cost")
    )
    len_cost_map = pl.DataFrame(
        {"n_char": range(1, len_target * 2)},
    ).with_columns([len_cost_col])
    px.line(len_cost_map, x="n_char", y="len_cost")
    return


@app.function
def len_cost(
    n_char: int,
    target: int,
    floor: int,
) -> float:
    return len_cost_3(n_char, target, floor)


@app.function
def len_cost_3(
    n_char: int,
    target: int,
    floor: int,
) -> float:
    if floor <= n_char <= target:
        return 0.0
    delta = n_char - target
    cost = abs(delta)
    cost = float(cost)
    if n_char < floor:
        short_penalty = ((target - n_char) / target) ** 2
        short_penalty *= target / floor
        short_penalty *= target
        cost += short_penalty
    return cost


@app.function
def len_cost_2(
    n_char: int,
    target: int,
    floor: int,
) -> float:
    rel = n_char / target
    short_penalty = 0
    if n_char < target:
        short_penalty = ((target - n_char) / target) ** 2
        short_penalty *= target / floor
        short_penalty *= target
    cost = (rel - 1) ** 2 + short_penalty
    return cost


@app.function
def len_cost_1(
    n_char: int,
    target: int,
    floor: int,
) -> float:
    delta = n_char - target
    cost = abs(delta)
    if n_char < floor:
        cost **= 2
    return cost


@app.function
def comp_dp(
    df: pl.DataFrame,
    len_target: int,
    len_floor: int,
    semantic_weight: float,
    split_penalty_base: float,
) -> tuple[list[int], np.ndarray, list[float], list[int]]:
    n_tokens = df.shape[0]
    n_bounds = n_tokens - 1
    cumsum = df.select("spacy_cumsum").to_series().to_list()
    n_char_total = cumsum[-1]

    segments = pl.DataFrame(range(n_tokens))
    segments = (
        segments.join(segments, how="cross")
        .rename({"column_0": "start", "column_0_right": "end"})
        .filter(pl.col("start") < pl.col("end"))
    )  #       ^ [0,n_bounds)   ^ [1,n_bounds+1]

    dp = [1e12] * n_tokens
    dp[0] = 0.0
    prev = [-1] * n_tokens
    jump = [0] * n_tokens
    transition = np.full((n_tokens, n_tokens), np.nan)

    for start, end in segments.iter_rows():
        n_char = cumsum[end]
        if start > 0:
            n_char -= cumsum[start]
        cost = len_cost(n_char, len_target, len_floor)
        if start > 0:
            semantic_cost = df.item(start, "norm_semantic_cost")
            cost += semantic_weight * semantic_cost
        cost += jump[start] * split_penalty_base / n_char_total
        cost += dp[start]
        transition[end][start] = cost
        if cost < dp[end]:
            dp[end] = cost
            prev[end] = start
            jump[end] = jump[start] + 1

    cur = n_bounds
    boundaries = []
    while prev[cur] > 0:
        boundaries.append(prev[cur])
        cur = prev[cur]
    boundaries.reverse()

    return boundaries, transition, dp, prev


@app.function(column=5)
def prepare(sent):
    if isinstance(sent, str):
        nlp = get_benepar_pipeline()
        doc = nlp(sent.strip())
        sent = list(doc.sents)[0]

    df = (
        pl.DataFrame(
            {
                "x": range(len(sent)),
                "distance": comp_distances(sent),
                "depth": comp_depths(sent),
                "spacy_tokens": [t.text for t in sent],
            }
        )
        .with_columns([pl.col("spacy_tokens").str.len_chars().alias("spacy_chars")])
        .with_columns([pl.col("spacy_chars").cum_sum().alias("spacy_cumsum")])
    )
    dist, depth = pl.col("distance"), pl.col("depth")
    m_dist, m_depth = dist.mean(), depth.mean()
    s_dist, s_depth = dist.std(), depth.std()
    df = df.with_columns(
        [
            (dist / s_dist).alias("norm_dist"),
            (depth / s_depth).alias("norm_depth"),
            ((dist - m_dist) / s_dist).alias("z_dist"),
            ((depth - m_depth) / s_depth).alias("z_depth"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("depth") - pl.col("distance")).alias("semantic_cost"),
            (pl.col("norm_depth") - pl.col("norm_dist")).alias("norm_semantic_cost"),
            (pl.col("z_depth") - pl.col("z_dist")).alias("z_semantic_cost"),
        ]
    )
    return df


@app.function
def comp_distances(sent):
    tree = ParentedTree.fromstring(sent._.parse_string)
    leaves = tree.leaves()
    n_leaves = len(leaves)
    n_tokens = len(sent)
    n_bounds = n_leaves - 1
    distances = []
    for i in range(n_bounds):
        path_l = tree.leaf_treeposition(i)
        path_r = tree.leaf_treeposition(i + 1)
        lca_depth = sum(1 for a, b in zip(path_l, path_r) if a == b)
        len_l = len(path_l) - lca_depth
        len_r = len(path_r) - lca_depth
        dist = len_l + len_r + 1
        distances.append(dist)
    leading_space = sent[0].text.strip() == ""
    leading_space &= n_leaves + 1 == n_tokens
    if leading_space:
        distances = [0] + distances
    return distances + [0]


@app.function
def comp_depths(sent):
    n_tokens = len(sent)
    n_bounds = n_tokens - 1
    depths = []
    for k in range(n_bounds):
        depth = 0
        for span in sent._.constituents:
            if span.start <= k + sent.start < span.end:
                depth += 1
        depths.append(depth)
    return depths + [0]


if __name__ == "__main__":
    app.run()
