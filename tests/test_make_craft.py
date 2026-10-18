"""Craft topology checks use synthetic parts rather than redistributing KSP craft."""
import re

import pytest

from orbiter.make_craft import (
    _attachment_vector, _part_id, _split_parts, _vector, make_terrier_craft,
)


def part(identifier, stage, position, *, parent=None, child=None,
         top=0.0, bottom=0.0, state='old_swivel_state = True'):
    fields = [
        f'\tpart = {identifier}', f'\tistg = {stage}', '\tdstg = 4',
        f'\tpos = 0,{position},0', '\tattPos0 = 0,-1,0',
    ]
    if parent:
        fields.append(f'\tattN = top,{parent}_0|{top}|0')
    if child:
        fields.extend([f'\tlink = {child}', f'\tattN = bottom,{child}_0|{bottom}|0'])
    return 'PART\n{\n' + '\n'.join(fields) + '\n\tEVENTS\n\t{\n\t}\n\tMODULE\n\t{\n\t\t' + state + '\n\t}\n}\n'


@pytest.fixture
def sources():
    blocks = [
        part('mk1pod.v2_1', -1, 7),
        part('fuelTankSmall_2', 1, 6.45525, child='liquidEngine2_3', bottom=-.55525),
        part('liquidEngine2_3', 2, 5, parent='fuelTankSmall_2',
             child='Decoupler.1_4', top=.9, bottom=-.7),
        part('Decoupler.1_4', 3, 4.25, parent='liquidEngine2_3',
             child='fuelTankSmall_5', top=.05, bottom=-.05),
        part('fuelTankSmall_5', 3, 3.64475, parent='Decoupler.1_4',
             child='liquidEngine2_6', top=.55525, bottom=-.55525),
        part('liquidEngine2_6', 5, 2.1895, parent='fuelTankSmall_5', top=.9),
    ]
    # The source conversion deliberately verifies Orbiter One's 21-part layout.
    # Twelve small synthetic attachments make the retained tree 18 parts.
    attachments = [f'basicFin_{i}' for i in range(7, 19)]
    blocks[4] = blocks[4].replace('\tEVENTS', ''.join(f'\tlink = {p}\n' for p in attachments) + '\tEVENTS')
    blocks.extend(part(identifier, 3, 3.5) for identifier in attachments)
    blocks.extend(part(identifier, 4, 0) for identifier in [
        'radialDecoupler_19', 'solidBooster.v2_20', 'noseCone_21',
    ])
    source = 'ship = Fixture\ndescription = Synthetic\nvesselType = Ship\n' + ''.join(blocks)
    template = part('liquidEngine3.v2_22', 2, 0, parent='fuelTankSmall_23',
                    child='Decoupler.1_24', top=0., bottom=-.8,
                    state='saved_terrier_setting = True')
    return source, 'ship = Template\n' + template


def test_terrier_swap_preserves_staging_modules_and_connected_geometry(sources):
    source, template = sources
    result = make_terrier_craft(source, template)
    prefix, blocks = _split_parts(result)
    parts = {_part_id(block): block for block in blocks}
    assert 'ship = OrbiterRL Terrier\n' in prefix
    assert len(parts) == 18
    assert 'liquidEngine2_3' not in result
    assert 'solidBooster' not in result
    upper = parts['liquidEngine3.v2_3']
    assert '\tistg = 2\n' in upper
    assert 'saved_terrier_setting = True' in upper
    assert 'old_swivel_state' not in upper
    assert '\tistg = 3\n' in parts['liquidEngine2_6']
    assert '\tistg = 2\n' in parts['Decoupler.1_4']
    # Both engine joints must remain attached at coincident world positions.
    engine_y = _vector(upper, 'pos')[1]
    assert engine_y + _attachment_vector(upper, 'top')[1] == pytest.approx(
        _vector(parts['fuelTankSmall_2'], 'pos')[1] + _attachment_vector(parts['fuelTankSmall_2'], 'bottom')[1])
    assert engine_y + _attachment_vector(upper, 'bottom')[1] == pytest.approx(
        _vector(parts['Decoupler.1_4'], 'pos')[1] + _attachment_vector(parts['Decoupler.1_4'], 'top')[1])
    # The shorter engine shifts every lower descendant, including radial fins.
    shift = .9 - .8 + .7
    assert _vector(parts['basicFin_7'], 'pos')[1] == pytest.approx(3.5 + shift)
    assert _vector(parts['liquidEngine2_6'], 'pos')[1] == pytest.approx(2.1895 + shift)
    assert _vector(parts['fuelTankSmall_2'], 'pos')[1] == pytest.approx(6.45525)


@pytest.mark.parametrize('count', [0, 2])
def test_terrier_template_requires_exactly_one_compatible_engine(sources, count):
    source, template = sources
    _, blocks = _split_parts(template)
    bad_template = 'ship = Invalid\n' + ''.join(
        block.replace('liquidEngine3.v2_22', f'liquidEngine3.v2_{i}')
        for i in range(count) for block in blocks)
    if count == 0:
        bad_template += part('basicFin_30', -1, 0)
    with pytest.raises(ValueError, match='exactly one liquidEngine3.v2'):
        make_terrier_craft(source, bad_template)


def test_terrier_conversion_rejects_incomplete_lower_attachment_tree(sources):
    source, template = sources
    source = re.sub(r'(?m)^\tlink = liquidEngine2_6$', '\tlink = missingPart_99', source)
    with pytest.raises(ValueError, match='Missing linked craft part'):
        make_terrier_craft(source, template)
