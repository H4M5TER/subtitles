from stable_whisper.whisper_compatibility import LANGUAGES
from typing import Annotated, Literal, TypeAlias
from pathlib import Path
import typer

LanguageKey = tuple(LANGUAGES.keys())
LanguageKeyType: TypeAlias = Literal[LanguageKey] | None  # type: ignore

ReadableFilePath = Annotated[
    Path,
    typer.Argument(
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
]
