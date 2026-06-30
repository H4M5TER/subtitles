import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    from marimo._runtime.output import append

    import re
    from pathlib import Path
    from functools import partial
    from copy import deepcopy
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
def get_benepar_pipeline(model_name: str) -> spacy.Language:
    dict_name = dict(
        benepar_en3_large="en_core_web_md",
        benepar_en3="en_core_web_md",
        benepar_zh2="zh_core_web_md",
    )[model_name]
    nlp = spacy.load(dict_name)
    nlp.add_pipe("benepar", config={"model": model_name})
    return nlp


@app.cell(column=1, hide_code=True)
def _():
    model_list = [
        "benepar_en3",
        "benepar_en3_large",
        "benepar_zh2",
    ]
    widget_model_name = mo.ui.dropdown(
        model_list,
        value=model_list[0],
        label="Select model",
    )

    widget_mult_cost = mo.ui.switch(
        value=False, label="Multiply split penalty by length penalty"
    )
    widget_len_target = mo.ui.slider(
        1, 100, 1, value=40, show_value=True, label="Target chars"
    )
    widget_short_scale = mo.ui.slider(
        1, 10, 1, value=3, show_value=True, label="Short split scale"
    )

    widget_semantic_weight = mo.ui.slider(
        0, 1000, 1, value=100, show_value=True, label="Semantic weight (%)"
    )
    widget_jump_weight = mo.ui.slider(
        0, 1000, 1, value=100, show_value=True, label="Jump penalty weight (%)"
    )
    widget_length_weight = mo.ui.slider(
        0, 1000, 1, value=100, show_value=True, label="Length weight (%)"
    )

    mo.vstack(
        [
            widget_model_name,
            mo.hstack([widget_mult_cost, widget_len_target, widget_short_scale]),
            mo.hstack([widget_semantic_weight, widget_length_weight, widget_jump_weight]),
        ]
    )
    return (
        widget_jump_weight,
        widget_len_target,
        widget_length_weight,
        widget_model_name,
        widget_mult_cost,
        widget_semantic_weight,
        widget_short_scale,
    )


@app.cell
def _(
    widget_jump_weight,
    widget_len_target,
    widget_length_weight,
    widget_model_name,
    widget_result_path,
    widget_semantic_weight,
    widget_text_input,
):
    use_input = widget_text_input.value.strip() != ""
    model_name = widget_model_name.value
    result_path = Path(widget_result_path.value)

    len_target = widget_len_target.value
    if model_name == "benepar_zh2":
        len_target //= 2
    semantic_weight = widget_semantic_weight.value / 100
    length_weight = widget_length_weight.value / 100
    jump_weight = widget_jump_weight.value / 100
    return (
        jump_weight,
        len_target,
        length_weight,
        model_name,
        result_path,
        semantic_weight,
        use_input,
    )


@app.cell
def _(
    model_name,
    result,
    sents,
    use_input,
    widget_segment_select,
    widget_text_input,
):
    if use_input:
        nlp = get_benepar_pipeline(model_name)
        doc = nlp(widget_text_input.value.strip())
        sent = list(doc.sents)[0]
    else:
        sent_id = widget_segment_select.value
        sent = sents[sent_id]
        segment = result[sent_id]
        whisper_tokens = [wt.word for wt in segment.words]

    n_tokens = len(sent)
    n_bounds = n_tokens - 1
    spacy_tokens = [t.text for t in sent]
    df = prepare(sent)
    df
    return df, segment, sent, spacy_tokens, whisper_tokens


@app.function
def prepare_batch(
    result_path: Path,
    model_name: str,
) -> tuple[WhisperResult, "Span"]:
    result = WhisperResult(str(result_path))
    result.reset()
    result.merge_all_segments()

    nlp = get_benepar_pipeline(model_name)
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
    # result.adjust_gaps(duration_threshold=0.5)
    return result, sents


@app.cell
def _(result, result_path, sents):
    regrouped_result = regroup(result, sents, 40)
    [s.text for s in regrouped_result.segments]
    regrouped_result.to_srt_vtt(
        str(result_path.with_suffix(".regroup.srt")), word_level=False
    )
    regrouped_result.save_as_json(str(result_path.with_suffix(".regroup.json")))
    return


@app.function
def regroup(
    result: WhisperResult,
    sents: "Span",
    len_target: int = 40,
):
    def split(
        result: WhisperResult,
        seg_index: int,
        word_index: int,
    ):
        segment = result[seg_index]
        sent = sents[seg_index]
        df = prepare(sent)
        spacy_bounds = comp_dp(df, len_target)[0]
        whisper_bounds = spacy_to_whisper(spacy_bounds, sent, segment)
        result.split_segment_by_index(segment, whisper_bounds, reassign_ids=False)
        return

    result = deepcopy(result)
    result.custom_operation("len=text", ">", len_target, split, word_level=False)
    result.reassign_ids()
    return result


@app.cell(column=2, hide_code=True)
def _():
    widget_result_path = mo.ui.text("")
    widget_text_input = mo.ui.text_area(
        "The prevailing view of scientific popularization, both within academic circles and beyond, affirms that its objectives and procedures are unrelated to tasks of cognitive development and that its pertinence is by and large restricted to the lay public. Consistent with this view, popularization is frequently portrayed as a logical and hence inescapable consequence of a culture dominated by science-based products and procedures and by a scientistic ideology. On another level, it is depicted as a quasi-political device for chan­ nelling the energies of the general public along predetermined paths; examples of this are the nineteenth-century Industrial Revolution and the U. S. -Soviet space race. Alternatively, scientific popularization is described as a carefully contrived plan which enables scientists or their spokesmen to allege that scientific learn­ ing is equitably shared by scientists and non-scientists alike. This manoeuvre is intended to weaken the claims of anti-scientific protesters that scientists monopolize knowledge as a means of sustaining their social privileges. Pop­ ularization is also sometimes presented as a psychological crutch. This, in an era of increasing scientific specialisation, permits the researchers involved to believe that by transcending the boundaries of their narrow fields, their endeavours assume a degree of general cognitive importance and even extra­ scientific relevance. Regardless of the particular thrust of these different analyses it is important to point out that all are predicated on the tacit presupposition that scientific popularization belongs essentially to the realm of non-science, or only concerns the periphery of scientific activity.",
        label="输入句子",
        full_width=True,
    )
    mo.vstack([widget_result_path, widget_text_input])
    return widget_result_path, widget_text_input


@app.cell(hide_code=True)
def _(model_name, result_path):
    if result_path.is_file():
        result, sents = prepare_batch(result_path, model_name)
        widget_segment_select = mo.ui.slider(
            0,
            len(result) - 1,
            1,
            debounce=True,
            show_value=True,
            label="Segment index",
        )
        widget_segment_select
    return result, sents, widget_segment_select


@app.cell(hide_code=True)
def _(segment, sent, spacy_bounds, use_input, whisper_tokens):
    mo.stop(use_input)

    whisper_bounds = spacy_to_whisper(spacy_bounds, sent, segment)
    whisper_segments = bounds_to_segments(
        whisper_tokens,
        whisper_bounds,
        left_closed=False,
    )
    whisper_segments
    # whisper_bounds
    return


@app.cell(hide_code=True)
def _(spacy_bounds, spacy_tokens):
    spacy_segments = bounds_to_segments(
        spacy_tokens,
        spacy_bounds,
        left_closed=True,
    )
    spacy_segments
    # spacy_bounds
    return


@app.function
def bounds_to_segments(
    tokens: list[str],
    bounds: list[int],
    left_closed: bool,
) -> list[str]:
    group = (
        pl.DataFrame(tokens)
        .with_row_index()
        .with_columns(
            pl.col("index")
            .cut(
                breaks=bounds,
                left_closed=left_closed,
            )
            .alias("group")
        )
        .group_by("group", maintain_order=True)
        .agg(pl.col("column_0").str.join(" ").alias("segment"))
    )
    return group.select("segment").to_series().to_list()


@app.function
def spacy_to_whisper(
    spacy_bounds: list[int],
    sent: "Span",
    segment: stable_whisper.Segment,
) -> list[int]:
    spacy_bounds = [i for i in spacy_bounds if 0 < i < len(sent)]
    char_len = [len(wordtiming.word) for wordtiming in segment.words]
    whisper_char_cumsum = np.cumsum(char_len)
    spacy_start_char = [
        sent[i].idx - sent.start_char
        #       ^ doc level
        for i in spacy_bounds
    ]
    whisper_bounds = [np.searchsorted(whisper_char_cumsum, i) for i in spacy_start_char]
    # remind: unsort indices make span overlapped
    # when split using stable_whisper
    return whisper_bounds


@app.cell(column=3, hide_code=True)
def _(bound_before_labels, colorscale, df, sorted_prev, transition):
    heatmap = go.Figure(
        layout=dict(
            width=800,
            height=800,
        )
    )

    transition_labels = bound_before_labels + ["EOS"]
    zeroed_transition = np.nan_to_num(transition)
    mirrored_transition = zeroed_transition + zeroed_transition.T
    np.fill_diagonal(mirrored_transition, np.nan)

    heatmap.add_trace(
        go.Heatmap(
            z=pl.from_numpy(mirrored_transition),
            x=transition_labels,
            y=transition_labels,
            colorscale=colorscale,
            colorbar=dict(orientation="h"),
        )
    )

    html = ""
    best_prev = [sorted_prev[i][0] for i in range(len(sorted_prev))]
    for i, bound in enumerate(sorted_prev[-1][:4]):
        if np.isnan(bound):
            break
        path = build_path(bound, best_prev)
        path += [-1]
        kth_segments = bounds_to_segments(
            df.select("spacy_tokens").to_series().to_list(),
            path,
            left_closed=True,
        )
        cost = transition[-1][bound]
        color = plotly.colors.carto.Vivid[i]
        separator = '<br><span style="color: red;"> | </span>'
        inner_html = separator.join(kth_segments)
        html += f'<div style="color: {color};">{inner_html}</div>'

        path = [transition_labels[i] for i in path]
        path_i = path[1:]
        path_j = path[:-1]
        heatmap.add_trace(
            go.Scatter(
                mode="lines+markers",
                name=f"第{i + 1}优路径",
                line=dict(color=color),
                marker=dict(size=20),
                x=path_i + [None] + path_j,
                y=path_j + [None] + path_i,
            )
        )

    mo.hstack([mo.vstack([mo.Html(html)]), heatmap])
    return


@app.cell
def _(transition):
    colors = plotly.colors.sequential.Blues_r
    quantiles = range(0, len(colors) * 10, 10)

    scales = np.nanpercentile(np.log(transition.flatten()), quantiles)
    scales = np.exp(scales)
    scales = np.interp(scales, (scales.min(), scales.max()), (0, 1))
    colorscale = list(zip(scales, colors))
    return (colorscale,)


@app.cell
def _(sent):
    append(sent._.parse_string)
    root = ParentedTree.fromstring(sent._.parse_string)
    append(root)  # need svgling
    return


@app.cell(column=4)
def _(
    df,
    jump_weight,
    len_target,
    length_weight,
    semantic_weight,
    widget_mult_cost,
    widget_short_scale,
):
    spacy_bounds, transition, sorted_prev = comp_dp(
        df,
        len_target,
        short_scale=widget_short_scale.value,
        mult_cost=widget_mult_cost.value,
        semantic_weight=semantic_weight,
        length_weight=length_weight,
        split_penalty_weight=jump_weight,
    )
    return sorted_prev, spacy_bounds, transition


@app.cell
def _(spacy_tokens):
    bound_before_labels = [f'{i},"{token}",' for i, token in enumerate(spacy_tokens)]
    return (bound_before_labels,)


@app.cell(hide_code=True)
def _(bound_before_labels, df):
    px.line(
        df,
        x=bound_before_labels,
        # y=[c for c in df.columns if c != "x"],
        y=[
            "dist_tree",
            "lca_depth",
            "semantic_reward",
        ],
    )
    return


@app.function
def comp_dp(
    df: pl.DataFrame,
    len_target: int,
    short_scale: int = 3,
    mult_cost: bool = False,
    semantic_weight: Optional[float] = 1,
    length_weight: Optional[float] = 1,
    split_penalty_weight: Optional[float] = 1,
):
    n_tokens = df.shape[0]
    n_bounds = n_tokens + 1  # 含首尾
    cumsum = df.select("spacy_cumsum").to_series().to_list()
    cumsum = [0] + cumsum  # 填充到 1-indexed 和 cur_split_bound 对齐
    n_char_total = cumsum[-1]
    semantic_rewards = df.select("semantic_reward").to_series().to_list()
    # 代表第 i 个边界(含首)的语义成本, 0-indexed
    # 或代表以第 i 个 token 之后作为分割边界的语义成本, 1-indexed
    split_penalty_weight *= len_target / n_char_total

    best = [1e12] * n_bounds
    best[0] = 0.0
    # 代表 token[0,i) 的最佳分割成本, 0-indexed
    # 从所有 transition 中取优
    prev = [-1] * n_bounds
    jumps = [0] * n_bounds
    transition = np.full((n_bounds, n_bounds), np.nan)

    for cur_split_bound in range(1, n_bounds):  # 不含首
        for last_split_bound in range(cur_split_bound):  # 不含尾
            jump = jumps[last_split_bound] + 1
            cost = split_penalty_weight * jump
            n_char = cumsum[cur_split_bound] - cumsum[last_split_bound]
            len_penalty = length_weight * len_cost(n_char, len_target, short_scale)
            if mult_cost and split_penalty_weight > 0:
                cost *= 1 + len_penalty
            else:
                cost = cost + len_penalty

            semantic_reward = semantic_weight * semantic_rewards[last_split_bound]
            # 注意这是上一个切分点的语义成本
            cost -= semantic_reward

            cost += best[last_split_bound]
            transition[cur_split_bound][last_split_bound] = cost
            if cost < best[cur_split_bound]:
                best[cur_split_bound] = cost
                prev[cur_split_bound] = last_split_bound
                jumps[cur_split_bound] = jump

    sorted_prev = [sort_cost(transition[i]) for i in range(len(transition))]
    best_prev = [sorted_prev[i][0] for i in range(len(sorted_prev))]
    boundaries = build_path(-1, best_prev)

    return boundaries, transition, sorted_prev


@app.function
def build_path(end: int, prev: list[int]) -> list[int]:
    cur = end
    path = [end]
    while prev[cur] >= 0:
        path.append(prev[cur])
        cur = prev[cur]
    path.reverse()
    return path


@app.function
def sort_cost(row: list[float]) -> list[int]:
    filtered = filter(lambda x: not np.isnan(x[1]), enumerate(row))
    sorted_ = sorted(filtered, key=lambda x: x[1])
    result = [i for i, v in sorted_]
    result += (len(row) - len(result)) * [np.nan]
    return result


@app.function(column=5)
def prepare(sent: "Span"):
    return (
        pl.DataFrame(
            {
                "x": range(len(sent)),
                "dist_tree": comp_dist_tree(sent),
                "lca_depth": comp_lca_depth(sent),
                "spacy_tokens": [t.text for t in sent],
            }
        )
        .with_columns([pl.col("spacy_tokens").str.len_chars().alias("spacy_chars")])
        .with_columns([pl.col("spacy_chars").cum_sum().alias("spacy_cumsum")])
        .with_columns(
            [
                pds.z_normalize(
                    pl.col("dist_tree")
                    # / (pl.col("lca_depth") + 4)
                    * (1 - minmax_normalize("lca_depth"))
                    # * (pds.z_normalize("lca_depth")) * -1
                ).alias("semantic_reward")
            ]
        )
    )


@app.function
def minmax_normalize(x: str | pl.Expr, l: float = 0, r: float = 1) -> pl.Expr:
    if isinstance(x, str):
        x = pl.col(x)
    min_ = x.min()
    max_ = x.max()
    res = (x - min_) / (max_ - min_)

    range = r - l
    res = res * range + l
    return res


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
        lca_depth = 0
        for a, b in zip(path_l, path_r):
            if a == b:
                lca_depth += 1
            else:
                break
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


@app.function
def comp_lca_depth(sent) -> list[int]:
    tree = ParentedTree.fromstring(sent._.parse_string)
    leaves = tree.leaves()
    # sent 以空格为第一个 token 时，tree.leaves 并不会包含空格
    # 因此少一个 token，导致 n_leaves = n_tokens + 1
    # 这其实是非常奇怪的，因为 tree 里仍然存在空格
    n_leaves = len(leaves)
    n_tokens = len(sent)
    n_bounds = n_leaves - 1
    depths = []
    for i in range(n_bounds):
        path_l = tree.leaf_treeposition(i)
        path_r = tree.leaf_treeposition(i + 1)
        lca_depth = 0
        for a, b in zip(path_l, path_r):
            if a == b:
                lca_depth += 1
            else:
                break
        depths.append(lca_depth)

    leading_space = sent[0].text.strip() == ""
    leading_space &= n_leaves + 1 == n_tokens
    if leading_space:
        depths = [0] + depths
    return [0] + depths


@app.function
def len_cost(
    n_char: int,
    target: int,
    short_scale: int = 3,
) -> float:
    relative = n_char / target - 1
    if relative < 0:
        relative *= short_scale
        # 提前在 (n-1)/n 处就达到原本峰值 1
        # 新峰值提高到 short_scale²
    cost = relative**2
    return cost


@app.cell(hide_code=True)
def _(len_target):
    range_n_char = range(0, max(len_target * 2 + 1, 100))
    len_cost_map = [len_cost(n_char, len_target) for n_char in range_n_char]
    px.line(x=range_n_char, y=len_cost_map)
    return


if __name__ == "__main__":
    app.run()
