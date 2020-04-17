from collections import defaultdict
from copy import deepcopy
from typing import Callable, Dict, List, Tuple

import numpy as np

is_x_in_y = lambda x, y: x[0] >= y[0] and x[1] <= y[1]


def break_and_collapse_sections(sentences: List[List[Tuple[int, int]]], min_len=100, max_len=400):
    new_sentences = []

    current_section = []
    current_length = 0
    for section in sentences :
        section_length = section[-1][1] - section[0][0]

        if current_length > max_len :
            new_sentences.append(current_section)
            current_section = []
            current_length = 0

        if section_length < min_len :
            current_section += section
            current_length += section_length

        else :
            new_sentences.append(current_section + section)
            current_section = []
            current_length = 0

    if len(current_section) > 0:
        new_sentences.append(current_section)

    assert [s for sec in new_sentences for s in sec] == [s for sec in sentences for s in sec], breakpoint()

    broken_sections = []
    for section in new_sentences :
        section_length = section[-1][1] - section[0][0]
        if section_length < max_len :
            broken_sections.append(section)
        else :
            current_section = []
            current_length = 0
            for sentence in section :
                sentence_length = sentence[1] - sentence[0]
                if current_length + sentence_length > max_len :
                    if len(current_section) > 0:
                        broken_sections.append(current_section)
                    current_section = [sentence]
                    current_length = sentence_length
                else :
                    current_section.append(sentence)
                    current_length += sentence_length

            if len(current_section) > 0:
                broken_sections.append(current_section)

    assert broken_sections[0][0] == sentences[0][0]
    assert broken_sections[-1][-1] == sentences[-1][-1]

    try :
        return [(x[0][0], x[-1][-1]) for x in broken_sections]
    except :
        breakpoint()


def collapse_paragraphs(plist, min_len=100, max_len=400):
    para_lengths = [p[1] - p[0] for p in plist]
    new_paragraphs = []
    cur_para_len = 0
    for p in para_lengths:
        if cur_para_len > max_len:
            new_paragraphs.append(cur_para_len)
            cur_para_len = 0
        if p < min_len:
            cur_para_len += p
        else:
            new_paragraphs.append(cur_para_len + p)
            cur_para_len = 0

    new_paragraphs.append(cur_para_len)
    assert sum(para_lengths) == sum(new_paragraphs), (sum(para_lengths), sum(new_paragraphs))
    return new_paragraphs


def break_paragraphs(plist, max_len=400):
    new_paragraphs = []
    for p in plist:
        if p < max_len:
            new_paragraphs.append(p)
        else:
            new_paragraphs += [max_len] * (p // max_len)
            if p % max_len > 0:
                new_paragraphs.append(p % max_len)

    assert sum(plist) == sum(new_paragraphs), (sum(plist), sum(new_paragraphs))
    return new_paragraphs


def move_boundaries(plist, elist):
    ends = np.cumsum(plist)
    starts = ends - np.array(plist)
    starts, ends = list(starts), list(ends)

    elist = sorted(elist, key=lambda x: (x[0], x[1]))
    para_stack = list(zip(starts, ends))
    new_paragraphs = []
    eix = 0
    while len(para_stack) > 0:
        p = para_stack.pop(0)

        while True:
            if eix >= len(elist):
                new_paragraphs.append(p)
                break
            elif elist[eix][0] >= p[0] and elist[eix][1] <= p[1]:
                eix += 1
            elif elist[eix][0] >= p[1]:
                new_paragraphs.append(p)
                break
            elif elist[eix][0] >= p[0]:
                p1 = para_stack.pop(0)
                new_paragraphs.append((p[0], elist[eix][1]))
                para_stack.insert(0, (elist[eix][1], p1[1]))
                eix += 1
                break

    assert new_paragraphs[0][0] == starts[0]
    assert new_paragraphs[-1][1] == ends[-1]
    for p, q in zip(new_paragraphs[:-1], new_paragraphs[1:]):
        assert p[1] == q[0]

    for e in elist:
        done = False
        for p in new_paragraphs:
            if is_x_in_y((e[0], e[1]), p):
                done = True
        assert done

    return new_paragraphs


def get_wastage(para_lengths):
    total_padded_input = [max(p) * len(p) for i, p in enumerate(para_lengths)]
    total_input = [sum(p) for i, p in enumerate(para_lengths)]
    return (sum(total_padded_input) - sum(total_input)) / sum(total_padded_input) * 100


def gen_lens(plist):
    return [p[1] - p[0] for p in plist]



