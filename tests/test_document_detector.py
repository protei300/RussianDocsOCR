"""Documents first (decision #142): grouping the detector's boxes and pairing two sides.

The detector has two classes - 'document' (whatever lies in the frame as one piece)
and 'page' (a page of an internal passport). A page belongs to the document it lies
in; documents come out largest first because process_img reads the first one.
process_frame reads every document and pairs the front and back of a vehicle
registration certificate by the number printed on both (issue #26).
"""
from document_processing.pipeline.pipeline import PipelineResults, pair_sides
from document_processing.pipeline_modules.document_detector import group_documents


def test_pages_belong_to_the_document_they_lie_in_and_documents_go_largest_first():
    boxes = [
        [500, 50, 700, 200, 0.9, 0, 'document'],        # a small card
        [0, 0, 400, 600, 0.95, 0, 'document'],          # a passport spread
        [10, 310, 390, 590, 0.9, 1, 'page'],            # its lower page
        [10, 10, 390, 290, 0.9, 1, 'page'],             # its upper page
        [900, 900, 950, 950, 0.5, 1, 'page'],           # a page in no document
    ]
    docs = group_documents(boxes)

    assert [d['box'] for d in docs] == [[0, 0, 400, 600], [500, 50, 700, 200]]
    assert [p['box'] for p in docs[0]['pages']] == [[10, 10, 390, 290], [10, 310, 390, 590]], 'top to bottom'
    assert docs[1]['pages'] == []


def _side(doctype, number):
    r = PipelineResults()
    r._meta_results['DocType'] = doctype
    r._meta_results['OCR'] = {'Licence_number': number} if number is not None else {}
    return r


def test_front_and_back_with_one_number_pair_up():
    docs = [_side('STS_2019', '99 87 786940'), _side('DL_2020', '99 12 345678'),
            _side('STSBACK_2019', '9987 786940')]
    pair_sides(docs)
    assert (docs[0].paired_with, docs[2].paired_with, docs[1].paired_with) == (2, 0, None)


def test_two_certificates_on_one_sheet_do_not_cross_pair():
    docs = [_side('STS_2019', '99 87 786940'), _side('STSBACK_2019', '99 80 895276'),
            _side('STS_2019', '99 80 895276'), _side('STSBACK_2019', '99 87 786940')]
    pair_sides(docs)
    assert [d.paired_with for d in docs] == [3, 2, 1, 0]


def test_an_ambiguous_or_unread_number_pairs_nothing():
    docs = [_side('STS_2019', '99 87 786940'), _side('STS_2019', '99 87 786940'),
            _side('STSBACK_2019', '99 87 786940'), _side('STSBACK_1996', None)]
    pair_sides(docs)
    assert [d.paired_with for d in docs] == [None, None, None, None]
