import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    from marimo._runtime.output import append

    import re
    from pathlib import Path
    from typing import Optional

    import math
    import numpy as np
    import polars as pl
    import polars_ds as pds
    from polars_ds.pipeline.transforms import scale

    import plotly
    import plotly.express as px
    import plotly.graph_objects as go
    from wigglystuff import GraphWidget

    import stable_whisper
    from stable_whisper import WhisperResult
    import spacy
    from spacy import displacy
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
    widget_length = mo.ui.number(0, 1000, value=100, step=1, label="Length weight (%)")

    mo.vstack(
        [
            mo.hstack([widget_len_target, widget_len_floor]),
            mo.hstack([widget_semantic, widget_length, widget_split_penalty]),
        ]
    )
    return (
        widget_len_floor,
        widget_len_target,
        widget_length,
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


@app.cell(hide_code=True)
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
        .with_columns(
            pl.col("index").cut(breaks=spacy_bounds, left_closed=True).alias("group")
        )
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
    whisper_bounds = [np.searchsorted(cumsum, i) for i in whisper_bounds]
    whisper_bounds = [i for i in whisper_bounds if 0 <= i < len(segment.words)]
    whisper_bounds = sorted(list(set(whisper_bounds)))
    # unsort indices break splitting by stable_whisper
    return whisper_bounds


@app.cell(column=3, hide_code=True)
def _(
    bound_before_labels,
    colorscale,
    colorscale2,
    transition,
    transition_best,
):
    layout = dict(
        width=500,
        height=500,
        autosize=False,
        yaxis_scaleanchor="x",
    )
    transition_labels = bound_before_labels + ["EOS"]

    heatmap1 = go.Figure(
        go.Heatmap(
            z=pl.from_numpy(transition),
            x=transition_labels,
            y=transition_labels,
            colorscale=colorscale,
        ),
        layout=layout,
    )
    heatmap2 = go.Figure(
        go.Heatmap(
            z=pl.from_numpy(transition_best),
            x=transition_labels,
            y=transition_labels,
            colorscale=colorscale2,
        ),
        layout=layout,
    )
    mo.hstack([heatmap1, heatmap2])
    return


@app.cell
def _(transition, transition_best):
    quantiles = [0, 1, 2, 3, 5, 10, 25, 50, 75, 100]
    colors = plotly.colors.sequential.Viridis

    scales = np.nanpercentile(transition.flatten(), quantiles)
    scales = np.interp(scales, (scales.min(), scales.max()), (0, 1))
    colorscale = list(zip(scales, colors))

    scales2 = np.nanpercentile(transition_best.flatten(), quantiles)
    scales2 = np.interp(scales2, (scales2.min(), scales2.max()), (0, 1))
    colorscale2 = list(zip(scales2, colors))
    # colorscale2 = colorscale
    return colorscale, colorscale2


@app.cell
def _(sent):
    append(sent._.parse_string)
    root = ParentedTree.fromstring(sent._.parse_string)
    append(root)  # need svgling
    root.pretty_print()
    return


@app.cell
def _(sent):
    mo.Html(displacy.render(sent, style="dep"))
    return


@app.cell(column=4, hide_code=True)
def _(
    widget_len_floor,
    widget_len_target,
    widget_length,
    widget_semantic,
    widget_split_penalty,
):
    len_target = widget_len_target.value
    len_floor = widget_len_floor.value

    semantic_weight = widget_semantic.value / 100
    length_weight = widget_length.value / 100
    split_penalty_weight = widget_split_penalty.value / 100
    return (
        len_floor,
        len_target,
        length_weight,
        semantic_weight,
        split_penalty_weight,
    )


@app.cell
def _(
    df,
    len_floor,
    len_target,
    length_weight,
    n_tokens,
    semantic_weight,
    split_penalty_weight,
):
    spacy_bounds, transition, dp, prev = comp_dp(
        df, len_target, len_floor, semantic_weight, length_weight, split_penalty_weight
    )
    transition_best = [
        [transition[i][j] if prev[j] == i else np.nan for j in range(n_tokens)]
        for i in range(n_tokens)
    ]
    transition_best = np.array(transition_best)
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


@app.function
def comp_dp(
    df: pl.DataFrame,
    len_target: int,
    len_floor: int,
    semantic_weight: Optional[float] = 1,
    length_weight: Optional[float] = 1,
    split_penalty_weight: Optional[float] = 1,
) -> tuple[list[int], np.ndarray, list[float], list[int]]:
    n_tokens = df.shape[0]
    n_bounds = n_tokens - 1
    cumsum = df.select("spacy_cumsum").to_series().to_list()
    n_char_total = cumsum[-1]
    semantic_costs = df.select("z_semantic_cost").to_series().to_list()
    # semantic_costs[i] 代表以第 i 个 token 之后作为分割边界的语义成本，1-indexed
    split_penalty_weight *= len_target / n_char_total

    best = [1e12] * n_tokens
    best[0] = 0.0
    # 有两种方式解释 best 的索引：
    # 1.1 best[i] 代表 token[0,i) 的最佳分割成本，0-indexed
    # 1.2 best[i] 代表以第 i 个 token 之前作为分割边界的最佳分割成本，0-indexed
    # 2.1 best[i] 代表 token[1,i] 的最佳分割成本，1-indexed
    # 2.2 best[i] 代表以第 i 个 token 之后作为分割边界的最佳分割成本，1-indexed
    prev = [-1] * n_tokens
    jumps = [0] * n_tokens
    transition = np.full((n_tokens, n_tokens), np.nan)

    for cur_split_bound in range(1, n_bounds + 1):  # [1,n_bounds + 1)
        for last_split_bound in range(cur_split_bound):  # [0,n_bounds)
            jump = jumps[last_split_bound] + 1
            cost = split_penalty_weight * jump
            n_char = cumsum[cur_split_bound - 1]
            if last_split_bound > 0:
                n_char -= cumsum[last_split_bound - 1]
            # 两个 split_bound 都是偏移了+1的分割边界
            # 边界 i 分割 token[i] 和 token[i+1]
            # cumsum 是 token 字符数的前缀和，和 token 的索引对齐
            # 因此偏移-1得到split_bound之前的字符数
            len_penalty = length_weight * len_cost(n_char, len_target)
            if False:
                cost *= 1 + len_penalty
            else:
                cost += len_penalty

            semantic_cost = semantic_weight * semantic_costs[last_split_bound]
            cost -= semantic_cost

            cost += best[last_split_bound]
            transition[last_split_bound][cur_split_bound] = cost
            if cost < best[cur_split_bound]:
                best[cur_split_bound] = cost
                prev[cur_split_bound] = last_split_bound
                jumps[cur_split_bound] = jump

    cur = n_bounds
    boundaries = []
    while prev[cur] > 0:
        boundaries.append(prev[cur])
        cur = prev[cur]
    boundaries.reverse()

    return boundaries, transition, best, prev


@app.function(column=5)
def prepare(sent):
    if isinstance(sent, str):
        nlp = get_benepar_pipeline()
        doc = nlp(sent.strip())
        sent = list(doc.sents)[0]

    n_tokens = len(sent)
    df = pl.DataFrame(
        {
            "x": range(n_tokens),
            "dist_tree": comp_dist_tree(sent),
            "dist_depth": comp_dist_depth(sent),
            "spacy_tokens": [t.text for t in sent],
        }
    )
    df = df.with_columns(
        [
            pl.col("spacy_tokens").str.len_chars().alias("spacy_chars"),
            pds.z_normalize("dist_tree").alias("z_dist_tree"),
            pds.z_normalize("dist_depth").alias("z_dist_depth"),
        ]
    )
    df = df.with_columns(
        [
            pl.col("spacy_chars").cum_sum().alias("spacy_cumsum"),
            (pl.col("dist_tree") + pl.col("dist_depth")).alias("semantic_cost"),
            (pl.col("z_dist_tree") + pl.col("z_dist_depth")).alias("z_semantic_cost"),
        ]
    )
    return df


@app.function
def comp_dist_tree(sent) -> list[int]:
    tree = ParentedTree.fromstring(sent._.parse_string)
    leaves = tree.leaves()
    # sent 以空格为第一个 token 时，tree.leaves 并不会包含空格
    # 因此少一个 token，导致 n_leaves = n_tokens + 1
    # 这其实是非常奇怪的，因为 tree 里仍然存在空格
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
        dist = len_l + len_r
        distances.append(dist)
    distances[-1] = 0  # 去除句号异常值

    leading_space = sent[0].text.strip() == ""
    leading_space &= n_leaves + 1 == n_tokens
    if leading_space:
        distances = [0] + distances
    return [0] + distances


@app.function
def comp_dist_depth(sent) -> list[int]:
    n_tokens = len(sent)
    depths = []
    for k in range(n_tokens):
        depth = 0
        for span in sent._.constituents:
            if span.start <= k + sent.start < span.end:
                depth += 1
        depths.append(depth)
    depths[-1] = depths[-2]  # 去除句号异常值
    distances = np.diff(depths, prepend=depths[0])
    # 填充一个位置，和动态规划的索引对齐，保持 distances[0] = 0
    return np.abs(distances)


@app.cell(hide_code=True)
def _(len_target):
    from functools import partial

    len_cost_col = (
        pl.col("n_char")
        .map_elements(
            partial(
                len_cost,
                target=len_target,
                # floor=len_floor,
            )
        )
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
) -> float:
    relative = n_char / target - 1
    if relative < 0:
        relative *= 2  # 提前在 1/2 处就达到 1
    cost = relative**2
    return cost


if __name__ == "__main__":
    app.run()
