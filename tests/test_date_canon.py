# -*- coding: utf-8 -*-
"""The canonical ``dd.mm.yyyy`` view of a recognised date.

Two questions live apart here on purpose: "was the date read correctly" is
measured on documents by the quality harness, "is the conversion correct" is
this file - a pure function over strings, so it can be shown a deliberately bad
input and be seen to refuse.

Both controls are present, and that is the point:

* POSITIVE - the material contains every form the documents actually print
  (worded months, the «Г.» and «ГОДА» tails, the 1997 passport's nominative
  «03.АВГУСТ.1989»), so a converter that quietly did nothing would fail here;
* NEGATIVE - dates the converter must REFUSE (no year, 31 February, garbage),
  so a converter that guessed would fail here too.
"""
import pytest

from document_processing.pipeline.dates import to_ddmmyyyy, canonical_dates

# Ровно то, что печатают документы - по одному представителю на форму.
PRINTED = [
    ('22.06.2010', '22.06.2010'),                 # цифровая, уже канон
    ('04.08.2016', '04.08.2016'),
    ('15 ОКТЯБРЯ 2020 Г.', '15.10.2020'),         # BIRTHCERT_2018, хвост «Г.»
    ('28 ИЮЛЯ 2010', '28.07.2010'),               # BIRTHCERT_1998, без хвоста
    ('10 ДЕКАБРЯ 1999 ГОДА', '10.12.1999'),       # SNILS, хвост «ГОДА»
    ('9 МАРТА 1993', '09.03.1993'),               # день без ведущего нуля
    ('03.АВГУСТ.1989', '03.08.1989'),             # INTPASSPORT_1997, именительный
    ('21 ИЮНЯ 1985', '21.06.1985'),
]

# Входы, на которых преобразование обязано ОТКАЗАТЬСЯ, а не выдумать.
REFUSED = [
    '5 МАЯ',              # нет года
    '15 ОКТЯБРЯ Г.',      # нет года, только хвост
    '31.02.2020',         # не существует в календаре
    '32.01.2020',         # дня 32 не бывает
    '15.13.2020',         # месяца 13 не бывает
    '15 ОКТЯБРЯ 20',      # двузначный год - 1920 или 2020? не угадываем
    'ОКТЯБРЯ',            # только месяц
    'КАКАЯ-ТО СТРОКА',    # не дата вовсе
    '',                   # пусто
    None,                 # ничего
]


@pytest.mark.parametrize('printed,canonical', PRINTED)
def test_printed_forms_convert(printed, canonical):
    assert to_ddmmyyyy(printed) == canonical


@pytest.mark.parametrize('text', REFUSED)
def test_converter_refuses_rather_than_guesses(text):
    assert to_ddmmyyyy(text) is None


def test_snils_wording_is_not_eaten():
    """The SNILS date is printed in words and reaches the result with «ГОДА».

    Its canonical form must exist, and the READING must stay untouched - the
    ground truth describes the image, and the accuracy measurement compares
    against the reading.
    """
    ocr = {'Birth_date': '26 СЕНТЯБРЯ 1997 ГОДА'}
    normalized = canonical_dates(ocr, ['Birth_date'])
    assert normalized == {'Birth_date': '26.09.1997'}
    assert ocr == {'Birth_date': '26 СЕНТЯБРЯ 1997 ГОДА'}


def test_only_converted_fields_appear():
    """A field the converter refuses is ABSENT, not empty and not the reading.

    That lets a consumer tell "there is no canonical form" from "the canonical
    form happens to equal the reading" - two different situations.
    """
    ocr = {'Birth_date': '15 ОКТЯБРЯ 2020 Г.',
           'Issue_date': '5 МАЯ',
           'Expiration_date': '',
           'Last_name_ru': 'ИВАНОВ'}
    normalized = canonical_dates(ocr, ['Birth_date', 'Issue_date', 'Expiration_date'])
    assert normalized == {'Birth_date': '15.10.2020'}


def test_non_date_fields_are_never_touched():
    """The caller passes the field list; nothing else is even looked at."""
    ocr = {'Licence_number': '62 1483828', 'Act_number': '110202778751843181007'}
    assert canonical_dates(ocr, ['Birth_date']) == {}
