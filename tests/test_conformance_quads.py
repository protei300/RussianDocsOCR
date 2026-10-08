"""The `quads` conformance stage is graded as coordinates.

Its numbers sit under field names (`quads.fields.Last_name_ru[0][1][0]`), so by the
generic leaf rule their tolerance would be looked up under the FIELD name and the GPU
profile's sub-pixel coordinate allowance would never reach them - the same trap the
positional bbox rows fell into (compare.py, BBOX_ROW_COLUMNS). No models needed.
"""
from conformance.runner.compare import CPU, GPU, _leaf, compare_json

QUAD = [[10.0, 20.0], [110.0, 20.0], [110.0, 40.0], [10.0, 40.0]]


def golden():
    return {'fields': {'Last_name_ru': [QUAD]}, 'words': {'Last_name_ru': [QUAD]}}


def moved(dx):
    q = [[x + dx, y] for x, y in QUAD]
    return {'fields': {'Last_name_ru': [q]}, 'words': {'Last_name_ru': [QUAD]}}


def test_every_quad_number_is_a_coordinate():
    assert _leaf('quads.fields.Last_name_ru[0][1][0]') == 'quad'
    assert _leaf('quads.address_lines[2][3][1]') == 'quad'


def test_cpu_is_strict_and_gpu_admits_a_sub_pixel_move():
    assert compare_json(golden(), moved(0.0005), CPU, path='quads') == []
    assert compare_json(golden(), moved(0.01), CPU, path='quads')
    assert compare_json(golden(), moved(0.5), GPU, path='quads') == []
    assert compare_json(golden(), moved(1.5), GPU, path='quads')


def test_unknown_way_back_differs_from_a_known_one():
    # `null` for the whole run (geometry.Unknown) is a different answer, not a pass
    assert compare_json({'fields': None, 'words': None}, golden(), CPU, path='quads')
