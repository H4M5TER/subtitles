import marimo

__generated_with = "0.23.6"
app = marimo.App()

with app.setup:
    import marimo as mo
    from yt_dlp import YoutubeDL
    import typer
    import json


@app.cell
def _(download, ydl_opts: dict):
    if mo.app_meta().mode == "edit":
        url = "https://www.youtube.com/watch?v="
        url += ""

        ydl_opts.update({"skip_download": True})
        download(url)
    return


@app.cell
def _():
    import logging

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    # ydl_opts["logger"] = logger
    # ydl_opts["verbose"] = True
    return


@app.cell
def _():
    ydl_opts: dict = {
        "outtmpl": "%(id)s",
        # jammed https://github.com/yt-dlp/yt-dlp/issues/7271
        # 'cookiesfrombrowser': ('vivaldi', 'default'),
        "cookiefile": "./cookies.txt",
        "writeinfojson": True,
        "writesubtitles": True,
        "allsubtitles": True,
        # "skip_download": True,
    }
    return (ydl_opts,)


@app.cell
def _(download):
    if mo.app_meta().mode == "script":
        typer.run(download)
    return


@app.cell
def _(ydl_opts: dict):
    def download(url: str):
        from os import makedirs
        from pathlib import Path

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            sanitized_info = ydl.sanitize_info(info)
        filtered_info = filter_info(sanitized_info)

        id = info["id"]
        download_path = Path("data/")
        makedirs(download_path, exist_ok=True)

        mo.output.append(filtered_info)
        with open(
            download_path / f"{id}.readable_info.json", "w+", encoding="utf-8"
        ) as f:
            json.dump(filtered_info, f, ensure_ascii=False, indent=2)
        if mo.app_meta().mode == "edit":
            mo.output.append(
                {
                    k: v
                    for k, v in info.items()
                    if k
                    not in (
                        "formats",
                        "requested_formats",
                        "thumbnails",
                        "automatic_captions",
                    )
                }
            )

        params = ydl_opts.copy()
        params["paths"] = {"home": str(download_path)}
        with YoutubeDL(params) as ydl:
            ydl.download([url])

    return (download,)


@app.function
def filter_info(info):
    from glom import glom, Coalesce

    result = glom(
        info,
        {
            "id": "id",
            "url": "webpage_url",
            "title": "title",
            "creator": lambda t: (
                t.get("creator") or t.get("uploader") + t.get("uploader_id")
            ),
            "upload_date": "upload_date",
            "description": "description",
            "tags": "tags",
            "chapters": Coalesce(("chapters", ["title"]), default=[]),
        },
    )
    return result


if __name__ == "__main__":
    app.run()
