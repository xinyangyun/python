"""Extract and normalize common Chinese time expressions from user input."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Callable


@dataclass(frozen=True)
class ExtractedTime:
    """A time expression found in the input text."""

    text: str
    start: int
    end: int
    value: datetime

    def to_dict(self) -> dict[str, str | int]:
        """Return a JSON-friendly representation."""
        result = asdict(self)
        result["value"] = self.value.isoformat(sep=" ")
        return result


_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_NUMBER = r"\d{1,4}|[零〇一二两三四五六七八九十]{1,5}"
_PERIOD = r"凌晨|早上|上午|中午|下午|傍晚|晚上"
_CLOCK = rf"""
    (?P<period>{_PERIOD})?
    (?P<hour>{_NUMBER})
    (?:
        [:\uff1a](?P<minute>\d{{1,2}})
        |
        点(?:(?P<half>半)|(?P<cn_minute>{_NUMBER})(?:分)?)?
    )
"""


def _parse_number(value: str) -> int:
    if value.isdigit():
        return int(value)
    if "十" not in value:
        digits = "".join(str(_CHINESE_DIGITS[char]) for char in value)
        return int(digits)

    tens, ones = value.split("十", maxsplit=1)
    tens_value = _CHINESE_DIGITS[tens] if tens else 1
    ones_value = _CHINESE_DIGITS[ones] if ones else 0
    return tens_value * 10 + ones_value


def _parse_clock(match: re.Match[str]) -> tuple[int, int]:
    hour = _parse_number(match.group("hour"))
    if match.group("half"):
        minute = 30
    elif match.group("minute"):
        minute = int(match.group("minute"))
    elif match.group("cn_minute"):
        minute = _parse_number(match.group("cn_minute"))
    else:
        minute = 0

    period = match.group("period")
    if period in {"下午", "傍晚", "晚上"} and hour < 12:
        hour += 12
    elif period == "中午" and 1 <= hour < 11:
        hour += 12
    elif period in {"凌晨", "早上", "上午"} and hour == 12:
        hour = 0

    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("时间超出有效范围")
    return hour, minute


def _with_clock(base: datetime, match: re.Match[str]) -> datetime:
    hour, minute = _parse_clock(match)
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _absolute_date(match: re.Match[str], reference: datetime) -> datetime:
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    base = datetime(year, month, day)
    return _with_clock(base, match) if match.group("hour") else base


def _month_day(match: re.Match[str], reference: datetime) -> datetime:
    month = int(match.group("month"))
    day = int(match.group("day"))
    base = datetime(reference.year, month, day)
    return _with_clock(base, match) if match.group("hour") else base


def _relative_day(match: re.Match[str], reference: datetime) -> datetime:
    offsets = {"大前天": -3, "前天": -2, "昨天": -1, "今天": 0, "明天": 1, "后天": 2, "大后天": 3}
    base = (reference + timedelta(days=offsets[match.group("relative_day")])).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return _with_clock(base, match) if match.group("hour") else base


def _weekday(match: re.Match[str], reference: datetime) -> datetime:
    weekday_text = match.group("weekday")
    target = 6 if weekday_text in {"日", "天"} else _parse_number(weekday_text) - 1
    prefix = match.group("week_prefix")

    if prefix == "下周":
        days = 7 - reference.weekday() + target
    elif prefix in {"本周", "这周"}:
        days = target - reference.weekday()
    else:
        days = (target - reference.weekday()) % 7

    base = (reference + timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    return _with_clock(base, match) if match.group("hour") else base


def _relative_offset(match: re.Match[str], reference: datetime) -> datetime:
    amount = _parse_number(match.group("amount"))
    unit = match.group("unit")
    delta = {
        "分钟": timedelta(minutes=amount),
        "小时": timedelta(hours=amount),
        "天": timedelta(days=amount),
    }[unit]
    return reference + delta if match.group("direction") == "后" else reference - delta


def _clock_only(match: re.Match[str], reference: datetime) -> datetime:
    return _with_clock(reference, match)


_PATTERNS: tuple[tuple[re.Pattern[str], Callable[[re.Match[str], datetime], datetime]], ...] = (
    (
        re.compile(
            rf"(?P<year>\d{{4}})[年/-](?P<month>\d{{1,2}})[月/-](?P<day>\d{{1,2}})日?"
            rf"(?:\s*{_CLOCK})?",
            re.VERBOSE,
        ),
        _absolute_date,
    ),
    (
        re.compile(
            rf"(?P<month>\d{{1,2}})月(?P<day>\d{{1,2}})[日号](?:\s*{_CLOCK})?",
            re.VERBOSE,
        ),
        _month_day,
    ),
    (
        re.compile(
            rf"(?P<relative_day>大前天|大后天|前天|昨天|今天|明天|后天)(?:\s*{_CLOCK})?",
            re.VERBOSE,
        ),
        _relative_day,
    ),
    (
        re.compile(
            rf"(?P<week_prefix>下周|本周|这周|周|星期)(?P<weekday>[一二三四五六日天])"
            rf"(?:\s*{_CLOCK})?",
            re.VERBOSE,
        ),
        _weekday,
    ),
    (
        re.compile(
            rf"(?P<amount>{_NUMBER})(?P<unit>分钟|小时|天)(?P<direction>后|前)",
            re.VERBOSE,
        ),
        _relative_offset,
    ),
    (re.compile(_CLOCK, re.VERBOSE), _clock_only),
)


def extract_times(text: str, reference_time: datetime | None = None) -> list[ExtractedTime]:
    """Extract time expressions and normalize them relative to ``reference_time``.

    Args:
        text: User input, for example ``"明天下午三点提醒我开会"``.
        reference_time: Base time for relative expressions. Defaults to now.

    Returns:
        Results ordered by their position in the input. Overlapping shorter matches
        are removed, so ``"明天下午三点"`` produces one result instead of two.
    """
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")

    reference = reference_time or datetime.now()
    candidates: list[ExtractedTime] = []
    for pattern, parser in _PATTERNS:
        for match in pattern.finditer(text):
            try:
                value = parser(match, reference)
            except (KeyError, TypeError, ValueError):
                continue
            candidates.append(
                ExtractedTime(
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    value=value,
                )
            )

    selected: list[ExtractedTime] = []
    for candidate in sorted(candidates, key=lambda item: (-(item.end - item.start), item.start)):
        overlaps = any(candidate.start < item.end and item.start < candidate.end for item in selected)
        if not overlaps:
            selected.append(candidate)
    return sorted(selected, key=lambda item: item.start)


if __name__ == "__main__":
    # sentence = "请在明天下午三点提醒我，后天早上9:30再跟进一次。"
    sentence = "设备编号 EQ-2026-0842 今天下午3点20分左右主轴异响，温度过高，报警码 ALM-403，拍了主轴照片和温度曲线图"
    for extracted in extract_times(sentence):
        print(extracted.to_dict())
