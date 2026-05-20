from __future__ import annotations

from pathlib import Path

import pytest
from pypulseq.utils.siemens.readasc import readasc


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8')


def test_readasc_parses_supported_value_types_and_indices(tmp_path: Path):
    asc_file = tmp_path / 'test.asc'

    _write_text(
        asc_file,
        '\n'.join(
            [
                '# comment line',  # ignored
                'a[0].b[2][3].c = "string" # trailing comment',  # string + indices
                'foo = 123',  # int
                'bar = 3.5',  # float
                'baz = -1.2e-3',  # scientific
                'hexval = 0x1A2b',  # hex
                '',  # empty line ignored
            ]
        )
        + '\n',
    )

    asc, extra = readasc(str(asc_file))

    assert extra == {}

    assert asc['a'][0]['b'][2][3]['c'] == 'string'
    assert asc['foo'] == 123
    assert asc['bar'] == 3.5
    assert asc['baz'] == -1.2e-3
    assert asc['hexval'] == int('1A2b', 16)


def test_readasc_splits_asc_and_extra_at_asconv_end(tmp_path: Path):
    asc_file = tmp_path / 'test.asc'

    _write_text(
        asc_file,
        '\n'.join(
            [
                'pre = 1',  # goes to asc
                '### ASCCONV END ###',
                'post = 2',  # goes to extra
            ]
        )
        + '\n',
    )

    asc, extra = readasc(str(asc_file))

    assert asc['pre'] == 1
    assert extra['post'] == 2


def test_readasc_raises_on_unparsed_assignment_line(tmp_path: Path):
    asc_file = tmp_path / 'test.asc'

    _write_text(
        asc_file,
        '\n'.join(
            [
                'bad = 1.2.3',  # contains '=' but does not match numeric regex
            ]
        )
        + '\n',
    )

    with pytest.raises(RuntimeError, match='ASC line with an assignment was not parsed correctly'):
        readasc(str(asc_file))


def test_readasc_raises_on_missing_file(tmp_path: Path):
    missing_file = tmp_path / 'does_not_exist.asc'

    with pytest.raises(FileNotFoundError):
        readasc(str(missing_file))
