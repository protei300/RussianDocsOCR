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

from document_processing.pipeline.dates import canonical_dates, record_date_to_ddmmyyyy, to_ddmmyyyy

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
    # BIRTHCERT_1998 печатает «10» ЯНВАРЯ 2013 г., рамка поля начинается на
    # кавычке, а «» нет в алфавите движка - он читает её ближайшей буквой
    # (issue #23). Одиночная буква вплотную к дню - прочитанная кавычка.
    ('И 10 ЯНВАРЯ 2013', '10.01.2013'),           # открывающая кавычка
    ('И10 ЯНВАРЯ 2013', '10.01.2013'),            # без пробела
    ('И 10 Н ЯНВАРЯ 2013', '10.01.2013'),         # обе кавычки
    ('10 П ЯНВАРЯ 2013 Г.', '10.01.2013'),        # закрывающая
    # Дата актовой записи: у формы 1998 обратный порядок и печатные слова между
    # частями, рамка их захватывает; у 2018 - обычный порядок.
    ('2010 ГОДА ИЮНЯ МЕСЯЦА 15 ЧИСЛА', '15.06.2010'),
    ('2010 года июня месяца 15', '15.06.2010'),   # «числа» за рамкой
    ('2 МАРТА 2025 Г.', '02.03.2025'),            # BIRTHCERT_2018
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
    # Буква, которая НЕ стоит вплотную к дню, и слово длиннее буквы - не
    # кавычка, а мусор: отказ, как и раньше.
    'ИЗ 10 ЯНВАРЯ 2013',  # слово, не одиночная буква
    '10 ЯНВАРЯ И 2013',   # буква не у дня, а у года
    'И 10 ЯНВАРЯ',        # кавычка отброшена, но года всё равно нет
    '2010 ГОДА ИЮНЯ МЕСЯЦА 31 ЧИСЛА',  # 31 июня не бывает
    '2010 ГОДА ИЮНЯ МЕСЯЦА ЧИСЛА',     # нет дня
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


# Дата актовой записи: порядок частей на бланке постоянный, поэтому её разбор
# находит части по виду - год, день, месяц внутри букв, - и переносит склейку
# слов и неверно прочитанные печатные «года»/«месяца».
RECORD_READ = [
    ('2015ГОДАИЮНЯИЕСЯЦА16', '16.06.2015'),       # склеено, «месяца» прочитано с ошибкой
    ('2026ТОДАМАЯНСЯЦА3', '03.05.2026'),
    ('2002ТОДАИЮНЯ,МЕСЯЦА18', '18.06.2002'),
    ('2003 ДЕКАБРЯ МЕСЯЦА 27', '27.12.2003'),
    ('2 МАРТА 2025 Г.', '02.03.2025'),            # обычный порядок берёт общий разбор
]
RECORD_REFUSED = [
    '2010 ГОДА ЦЮЛЯ МЕСЯЦА 17',   # сам месяц прочитан неверно
    '2020 ГОДА ИЮЛЯ МЕСЯЦА',      # нет дня
    '2020 ЯНВАРЯ 110201114335',   # в рамку попал номер записи
    '.110266032',
    '2010 ИЮНЯ МАЯ 15',           # два месяца
    '2010ГОДАИЮНЯ 15 16',         # два дня
]


@pytest.mark.parametrize('printed,canonical', RECORD_READ)
def test_record_date_reads_through_glue_and_misread_words(printed, canonical):
    assert record_date_to_ddmmyyyy(printed) == canonical


@pytest.mark.parametrize('text', RECORD_REFUSED)
def test_record_date_still_refuses_rather_than_guesses(text):
    assert record_date_to_ddmmyyyy(text) is None


def test_only_the_record_date_gets_the_lenient_reading():
    ocr = {'Act_date': '2015ГОДАИЮНЯИЕСЯЦА16', 'Issue_date': '2015ГОДАИЮНЯИЕСЯЦА16'}
    assert canonical_dates(ocr, ['Act_date', 'Issue_date']) == {'Act_date': '16.06.2015'}
