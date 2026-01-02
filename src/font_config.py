from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.font_manager as fm


def _add_font_files(font_files: Iterable[str]) -> None:
    for font_file in font_files:
        try:
            if Path(font_file).exists():
                fm.fontManager.addfont(font_file)
        except Exception:
            # Best-effort: if a font file is unreadable or unsupported, skip it.
            continue


def configure_matplotlib_chinese_fonts() -> list[str]:
    """
    Best-effort cross-platform Chinese font configuration for Matplotlib.

    - macOS: PingFang SC / Hiragino / Heiti / Songti
    - Windows: Microsoft YaHei / SimHei / SimSun
    - Linux: Noto CJK / WenQuanYi
    """
    system = platform.system().lower()

    # Try to register common font files (when present) so Matplotlib can find them.
    _add_font_files(
        [
            # macOS
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Supplemental/PingFang.ttc",
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "/System/Library/Fonts/Supplemental/STHeiti Light.ttc",
            "/System/Library/Fonts/Supplemental/STHeiti Medium.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Songti.ttc",
            # Linux (Noto CJK)
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
            # Linux (WenQuanYi)
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            # Windows (common locations)
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "msyh.ttc"),
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "msyhbd.ttc"),
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "simhei.ttf"),
            os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "simsun.ttc"),
        ]
    )

    mac_families = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Songti TC",
        "Arial Unicode MS",
    ]
    windows_families = [
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "SimHei",
        "SimSun",
        "NSimSun",
        "Arial Unicode MS",
    ]
    linux_families = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]
    common_tail = ["sans-serif"]

    if system == "darwin":
        font_families = mac_families + linux_families + windows_families + common_tail
    elif system == "windows":
        font_families = windows_families + mac_families + linux_families + common_tail
    else:
        font_families = linux_families + windows_families + mac_families + common_tail

    mpl.rcParams["font.sans-serif"] = font_families
    mpl.rcParams["axes.unicode_minus"] = False
    return font_families

