"""Split Chinese TTS input at natural boundaries without discarding text."""

import os

from lib.conf import models_dir
from lib.conf_lang import punctuation_split_hard_set, punctuation_split_soft_set
from lib.conf_models import SML_TAG_PATTERN


_CLOSING = frozenset('”’」』）》】)]\"\'')
_SOFT = (punctuation_split_soft_set | set('；：')) - {'·'}


def _units(text):
    """Keep an entire SML tag as one unit, with no spoken character cost."""
    result = []
    index = 0
    while index < len(text):
        match = SML_TAG_PATTERN.match(text, index) if text[index] == '[' else None
        if match:
            result.append(match.group())
            index = match.end()
        else:
            result.append(text[index])
            index += 1
    return result


def _numeric_punctuation(units, index):
    return (units[index] in {'.', ','} and 0 < index < len(units) - 1
            and units[index - 1].isdigit() and units[index + 1].isdigit())


def _boundary_end(units, index):
    end = index + 1
    while end < len(units) and (len(units[end]) > 1 or units[end] in _CLOSING
                                or units[end] in punctuation_split_hard_set):
        if _numeric_punctuation(units, end):
            break
        end += 1
    return end


def _word_boundary(units, limit):
    # Tokenize the full remainder so a word crossing the limit is not treated
    # as a complete word. SML placeholders retain the unit offsets.
    import jieba
    jieba.dt.cache_file = os.path.join(models_dir, 'jieba.cache')
    text = ''.join(unit if len(unit) == 1 else '\ue000' for unit in units)
    end = 0
    for _, _, word_end in jieba.tokenize(text):
        if word_end > limit:
            break
        end = word_end
    if not any(unit.isalnum() for unit in units[:end] if len(unit) == 1):
        return limit
    return end


def _bounded_chunks(units, max_chars):
    while units:
        count = 0
        limit = len(units)
        for index, unit in enumerate(units):
            count += int(len(unit) == 1)
            if count > max_chars:
                limit = index
                break
        if limit == len(units):
            yield ''.join(units).strip()
            return
        end = 0
        for index in range(limit):
            if units[index] in _SOFT and not _numeric_punctuation(units, index):
                candidate = _boundary_end(units, index)
                if candidate <= limit:
                    end = candidate
        if not end:
            end = _word_boundary(units, limit)
        if end > 1 and not any(unit.isalnum() for unit in units[end:] if len(unit) == 1):
            # Leave the last word with its closing punctuation instead of
            # creating a punctuation-only TTS call at the length limit.
            end = _word_boundary(units, end - 1)
        yield ''.join(units[:end]).strip()
        units = units[end:]


def split_chinese_sentences(text, max_chars):
    """Keep complete sentences; split oversized ones at clauses, then words."""
    if max_chars < 1:
        raise ValueError('max_chars must be positive')
    units = _units(text)
    spans = []
    start = 0
    index = 0
    while index < len(units):
        if units[index] in punctuation_split_hard_set and not _numeric_punctuation(units, index):
            end = _boundary_end(units, index)
            span = units[start:end]
            spoken = ''.join(unit for unit in span if len(unit) == 1)
            if any(char.isalnum() for char in spoken):
                spans.append(span)
                start = end
            index = end
        else:
            index += 1
    if start < len(units):
        tail = units[start:]
        if spans and not any(unit.isalnum() for unit in tail if len(unit) == 1):
            spans[-1].extend(tail)
        else:
            spans.append(tail)
    return [chunk for span in spans for chunk in _bounded_chunks(span, max_chars) if chunk]
