"""Craft topology checks use synthetic parts rather than redistributing KSP craft."""
import re

import pytest

from orbiter.make_craft import (
    _attachment_peer, _attachment_vector, _part_id, _split_parts, _vector,
    make_terrier_craft, make_transfer_craft,
)


def part(identifier, stage, position, *, parent=None, child=None,
         top=0.0, bottom=0.0, state='old_swivel_state = True'):
    fields = [
        f'\tpart = {identifier}', f'\tistg = {stage}', '\tdstg = 4',
        f'\tpersistentId = {1000+int(identifier.rsplit("_", 1)[1])}',
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


@pytest.fixture
def terrier_source():
    # Synthetic stock-size stack with the measured Terrier craft's 18-part
    # topology, eight FL-T200 tanks, and unchanged crew pod/parachute payload.
    specs = [('mk1pod.v2_1', -1, .642, -.405), ('Decoupler.1_2', 1, .05, -.05)]
    specs += [(f'fuelTankSmall_{i}', 1, .55525, -.55525) for i in (3, 4)]
    specs += [('liquidEngine3.v2_5', 2, 0., -.8), ('Decoupler.1_6', 2, .05, -.05)]
    specs += [(f'fuelTankSmall_{i}', 1, .55525, -.55525) for i in range(7, 13)]
    specs += [('liquidEngine2_13', 3, .9, -.7)]
    blocks, position = [], 20.
    for index, (identifier, stage, top, bottom) in enumerate(specs):
        parent = specs[index-1][0] if index else None
        child = specs[index+1][0] if index+1 < len(specs) else None
        relative = 0.
        if index:
            relative = specs[index-1][3]-top
            position += relative
        block = part(identifier, stage, position, parent=parent, child=child, top=top, bottom=bottom)
        block = re.sub(r'^\tattPos0 = [^\n]+$', f'\tattPos0 = 0,{relative},0', block, flags=re.MULTILINE)
        if identifier.startswith('fuelTankSmall_'):
            resource = ''.join(f'\tRESOURCE\n\t{{\n\t\tname = {name}\n\t\tamount = {amount}\n\t\tmaxAmount = {amount}\n\t}}\n'
                               for name, amount in [('LiquidFuel', 90), ('Oxidizer', 110)])
            block = block[:-2]+resource+'}\n'
        if identifier == 'mk1pod.v2_1':
            block = block.replace('\tEVENTS', '\tlink = parachuteSingle_18\n\tEVENTS')
        if identifier == 'fuelTankSmall_12':
            block = block.replace('\tEVENTS', ''.join(f'\tlink = basicFin_{i}\n' for i in range(14, 18))+'\tEVENTS')
        blocks.append(block)
    blocks.extend(part(f'basicFin_{i}', -1, 8.) for i in range(14, 18))
    blocks.append(part('parachuteSingle_18', 0, 20.7))
    return 'ship = OrbiterRL Terrier\ndescription = Synthetic\nsize = 3,14,2\n'+''.join(blocks)


def test_transfer_tanks_preserve_unique_tree_and_coincident_stack_joints(terrier_source):
    result = make_transfer_craft(terrier_source)
    prefix, blocks = _split_parts(result)
    parts = {_part_id(block): block for block in blocks}
    original = {_part_id(block): block for block in _split_parts(terrier_source)[1]}
    assert 'ship = OrbiterRL Transfer\n' in prefix
    assert len(blocks) == len(parts) == 20
    assert len(set(re.findall(r'^\tpersistentId = (\d+)$', result, re.MULTILINE))) == 20
    new_ids = set(parts)-set(original)
    assert len(new_ids) == 2 and all(name.startswith('fuelTankSmall_') for name in new_ids)
    parents = {}
    for identifier, block in parts.items():
        for child in re.findall(r'^\tlink = (\S+)', block, re.MULTILINE):
            assert child in parts and child not in parents
            parents[child] = identifier
        if '\tattN = bottom,' in block:
            child = _attachment_peer(block, 'bottom')
            assert _attachment_peer(parts[child], 'top') == identifier
            assert parents[child] == identifier
            for coordinate in range(3):
                assert _vector(block, 'pos')[coordinate]+_attachment_vector(block, 'bottom')[coordinate] == pytest.approx(
                    _vector(parts[child], 'pos')[coordinate]+_attachment_vector(parts[child], 'top')[coordinate])
    assert set(parts)-set(parents) == {'mk1pod.v2_1'}
    visited, pending = set(), ['mk1pod.v2_1']
    while pending:
        identifier = pending.pop()
        assert identifier not in visited
        visited.add(identifier)
        pending.extend(re.findall(r'^\tlink = (\S+)', parts[identifier], re.MULTILINE))
    assert visited == set(parts)
    assert parts['mk1pod.v2_1'] == original['mk1pod.v2_1']
    assert _vector(parts['fuelTankSmall_4'], 'pos') == _vector(original['fuelTankSmall_4'], 'pos')
    assert _vector(parts['liquidEngine3.v2_5'], 'pos')[1] == pytest.approx(
        _vector(original['liquidEngine3.v2_5'], 'pos')[1]-2.221)
    assert _vector(parts['basicFin_14'], 'pos')[1] == pytest.approx(8.-2.221)
    for identifier in new_ids | {'liquidEngine3.v2_5'}:
        parent = _attachment_peer(parts[identifier], 'top')
        assert _vector(parts[identifier], 'attPos0') == pytest.approx(tuple(
            a-b for a, b in zip(_vector(parts[identifier], 'pos'), _vector(parts[parent], 'pos'))))
    assert '\tistg = 3\n' in parts['liquidEngine2_13']
    assert '\tistg = 2\n' in parts['liquidEngine3.v2_5']
    assert 'size = 3,16.221,2\n' in prefix


def test_transfer_stock_tank_count_matches_requested_mass_profile(terrier_source):
    _, blocks = _split_parts(make_transfer_craft(terrier_source))
    parts = {_part_id(block): block for block in blocks}
    def subtree(identifier):
        found, pending = set(), [identifier]
        while pending:
            item = pending.pop()
            assert item not in found
            found.add(item)
            pending.extend(re.findall(r'^\tlink = (\S+)', parts[item], re.MULTILINE))
        return found
    lower = subtree('Decoupler.1_6')
    upper = subtree('Decoupler.1_2')-lower
    payload = set(parts)-lower-upper
    dry = {'mk1pod.v2': 800., 'Decoupler.1': 40., 'fuelTankSmall': 125.,
           'liquidEngine3.v2': 500., 'liquidEngine2': 1500., 'basicFin': 10., 'parachuteSingle': 100.}
    dry_mass = lambda ids: sum(dry[identifier.rsplit('_', 1)[0]] for identifier in ids)
    fuel_mass = lambda ids: 1000.*sum(identifier.startswith('fuelTankSmall_') for identifier in ids)
    assert (dry_mass(upper), fuel_mass(upper)) == (1040., 4000.)
    assert (dry_mass(lower), fuel_mass(lower)) == (2330., 6000.)
    assert dry_mass(payload)+40. == 940.  # Existing ten monopropellant units.
    assert dry_mass(set(parts))+fuel_mass(set(parts))+40. == 14310.


def test_transfer_rejects_empty_tank_and_duplicate_persistent_ids(terrier_source):
    with pytest.raises(ValueError, match='full stock FL-T200'):
        make_transfer_craft(terrier_source.replace('\t\tamount = 90', '\t\tamount = 0'))
    with pytest.raises(ValueError, match='unique persistentId'):
        make_transfer_craft(terrier_source.replace('persistentId = 1005', 'persistentId = 1004'))
