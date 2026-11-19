"""Post-search evidence only: read the published aggregate table, never demo tracks."""
import re


def aggregate_rows(text):
    sectors = {}
    for line in text.splitlines():
        cells = [v.strip() for v in line.strip().strip('|').split('|')]
        if not cells[0].isdigit(): continue
        if len(cells)!=9 or not re.fullmatch(r'\d+/\d+/\d+',cells[5]):
            raise ValueError('unexpected aggregate table schema')
        passed,died,censored=map(int,cells[5].split('/'))
        if passed+died+censored != int(cells[3]): raise ValueError('inconsistent aggregate counts')
        sectors.setdefault(cells[0],[]).append(dict(family=cells[1],materials=cells[2],passed=passed,died=died,censored=censored))
    return sectors
