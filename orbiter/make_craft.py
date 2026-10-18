"""Create conventional two-liquid-stage rockets from local stock craft files.

The installed stock craft is read, never modified or redistributed. Its single
radial solid booster is removed and the liquid interstage ignition/separation
is combined. An optional stock Terrier template replaces the heavy upper
Swivel, including its saved modules and shorter attachment geometry. Output
belongs in the dedicated OrbiterRL save's Ships/VAB folder.
"""
import argparse
from pathlib import Path
import re


def _split_parts(source):
    starts = list(re.finditer(r'^PART\s*\n\{', source, re.MULTILINE))
    if not starts:
        raise ValueError('No PART blocks found in craft')
    return source[:starts[0].start()], [
        source[m.start():starts[i+1].start() if i+1 < len(starts) else len(source)]
        for i, m in enumerate(starts)
    ]


def _part_id(block):
    return re.search(r'^\tpart = (\S+)', block, re.MULTILINE).group(1)


def _vector(block, key):
    match = re.search(rf'^\t{key} = ([^\n]+)', block, re.MULTILINE)
    if not match:
        raise ValueError(f'Missing {key} in craft part')
    return tuple(float(value) for value in match.group(1).split(','))


def _set_vector(block, key, value):
    result, count = re.subn(rf'^\t{key} = [^\n]+',
                            f'\t{key} = ' + ','.join(f'{v:.10g}' for v in value),
                            block, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f'Expected one {key} in craft part')
    return result


def _attachment_vector(block, node):
    match = re.search(rf'^\tattN = {node},([^\n]+)', block, re.MULTILINE)
    if not match:
        raise ValueError(f'Missing {node} attachment in craft part')
    return tuple(float(v) for v in match.group(1).rsplit('_', 1)[1].split('|'))


def make_liquid_craft(source: str) -> str:
    starts = list(re.finditer(r'^PART\s*\n\{', source, re.MULTILINE))
    if len(starts) != 21:
        raise ValueError('Expected the stock 21-part Orbiter One; source layout differs')
    prefix = source[:starts[0].start()]
    prefix = re.sub(r'^ship = .*$', 'ship = OrbiterRL Liquid', prefix, flags=re.MULTILINE)
    prefix = re.sub(r'^description = .*$', 'description = Two liquid stages for reproducible orbital-launch experiments. Derived locally from stock Orbiter One.', prefix, flags=re.MULTILINE)
    prefix = re.sub(r'^vesselType = .*$', 'vesselType = Ship', prefix, flags=re.MULTILINE)
    blocks = [source[m.start():starts[i+1].start() if i+1 < len(starts) else len(source)] for i, m in enumerate(starts)]
    ids = [re.search(r'^\s*part = (\S+)', b, re.MULTILINE).group(1) for b in blocks]
    removed = {p for p in ids if p.split('_')[0] in {'radialDecoupler', 'solidBooster.v2', 'noseCone'}}
    if len(removed) != 3:
        raise ValueError('Expected exactly one radial decoupler, solid booster, nose cone')
    kept = []
    for identifier, block in zip(ids, blocks):
        if identifier in removed:
            continue
        # Remove only top-level attachment/link lines mentioning deleted IDs.
        block = '\n'.join(line for line in block.split('\n') if not (
            re.match(r'^\t(?:link|attN|srfN) = ', line) and any(x in line for x in removed)))
        part_name = identifier.rsplit('_', 1)[0]
        old_stage = int(re.search(r'^\tistg = (-?\d+)', block, re.MULTILINE).group(1))
        old_dstg = int(re.search(r'^\tdstg = (-?\d+)', block, re.MULTILINE).group(1))
        new_stage = 3 if part_name == 'liquidEngine2' and old_stage == 5 else 2 if old_stage == 3 else old_stage
        new_dstg = 3 if old_dstg == 4 else 2 if old_dstg == 3 else old_dstg
        block = re.sub(r'^\tistg = -?\d+', f'\tistg = {new_stage}', block, flags=re.MULTILINE)
        block = re.sub(r'^\tdstg = -?\d+', f'\tdstg = {new_dstg}', block, flags=re.MULTILINE)
        kept.append(block)
    result = prefix + ''.join(kept)
    if any(identifier in result for identifier in removed):
        raise ValueError('A reference to removed booster parts remains')
    return result


def make_terrier_craft(source: str, terrier_source: str) -> str:
    """Use a stock Terrier part snapshot without modifying installed originals.

    Orbiter One's stack is vertical; both engines have their thrust axis along
    local Y. Keep its part IDs, stage plan, and attachments while transplanting
    the stock Terrier's module state. Translate the entire lower subtree to
    keep both engine stack joints coincident after the shorter engine swap.
    """
    prefix, blocks = _split_parts(make_liquid_craft(source))
    _, templates = _split_parts(terrier_source)
    terriers = [b for b in templates if _part_id(b).rsplit('_', 1)[0] == 'liquidEngine3.v2']
    if len(terriers) != 1:
        raise ValueError('Terrier template must contain exactly one liquidEngine3.v2')
    template = terriers[0]
    uppers = [i for i, b in enumerate(blocks)
              if _part_id(b).startswith('liquidEngine2_') and '\tistg = 2\n' in b]
    if len(uppers) != 1:
        raise ValueError('Expected exactly one stage-2 upper Swivel')
    index = uppers[0]
    old = blocks[index]
    old_id = _part_id(old)
    new_id = 'liquidEngine3.v2_' + old_id.rsplit('_', 1)[1]
    old_top, old_bottom = _attachment_vector(old, 'top'), _attachment_vector(old, 'bottom')
    new_top, new_bottom = _attachment_vector(template, 'top'), _attachment_vector(template, 'bottom')
    if any(abs(v) > 1e-5 for v in (*old_top[::2], *old_bottom[::2],
                                  *new_top[::2], *new_bottom[::2])):
        raise ValueError('Only coaxial vertical stack attachments are supported')
    engine_shift = old_top[1] - new_top[1]
    lower_shift = engine_shift + new_bottom[1] - old_bottom[1]
    links = re.findall(r'^\tlink = (\S+)', old, re.MULTILINE)
    if len(links) != 1:
        raise ValueError('Expected the upper engine to have one lower decoupler')
    lower_id = links[0]
    parts = {_part_id(b): b for b in blocks}
    descendants = set()
    pending = [lower_id]
    while pending:
        identifier = pending.pop()
        if identifier in descendants:
            raise ValueError('Cycle or repeated child in craft attachment tree')
        descendants.add(identifier)
        if identifier not in parts:
            raise ValueError(f'Missing linked craft part {identifier}')
        pending.extend(re.findall(r'^\tlink = (\S+)', parts[identifier], re.MULTILINE))
    # Module state must belong to the replacement engine, not the old Swivel.
    state_marker = '\tEVENTS\n'
    if state_marker not in old or state_marker not in template:
        raise ValueError('Missing stock part state section')
    replacement = old.split(state_marker, 1)[0] + state_marker + template.split(state_marker, 1)[1]
    replacement = replacement.replace(f'\tpart = {old_id}\n', f'\tpart = {new_id}\n')
    position = list(_vector(replacement, 'pos'))
    position[1] += engine_shift
    replacement = _set_vector(replacement, 'pos', position)
    attachment = list(_vector(replacement, 'attPos0'))
    attachment[1] += engine_shift
    replacement = _set_vector(replacement, 'attPos0', attachment)
    for node, vector in [('top', new_top), ('bottom', new_bottom)]:
        coords = '|'.join(f'{v:.10g}' for v in vector)
        replacement = re.sub(rf'^(\tattN = {node},[^\n]+_)[^_\n]+$',
                             lambda m: m.group(1) + coords, replacement, flags=re.MULTILINE)
    blocks[index] = replacement
    for i, block in enumerate(blocks):
        identifier = _part_id(block)
        if identifier in descendants:
            position = list(_vector(block, 'pos'))
            position[1] += lower_shift
            block = _set_vector(block, 'pos', position)
            if identifier == lower_id:
                attachment = list(_vector(block, 'attPos0'))
                attachment[1] += lower_shift - engine_shift
                block = _set_vector(block, 'attPos0', attachment)
        blocks[i] = block.replace(old_id, new_id)
    prefix = re.sub(r'^ship = .*$', 'ship = OrbiterRL Terrier', prefix, flags=re.MULTILINE)
    prefix = re.sub(r'^description = .*$',
                    'description = Sequential Swivel and Terrier liquid stages for orbital policy validation. Derived locally from stock craft.',
                    prefix, flags=re.MULTILINE)
    return prefix + ''.join(blocks)


def _attachment_peer(block, node):
    match = re.search(rf'^\tattN = {node},([^\n]+)', block, re.MULTILINE)
    if not match:
        raise ValueError(f'Missing {node} attachment in craft part')
    return match.group(1).rsplit('_', 1)[0]


def _set_attachment_peer(block, node, peer):
    result, count = re.subn(rf'^(\tattN = {node},)[^\n]+_([^_\n]+)$',
                            lambda match: match.group(1)+peer+'_'+match.group(2),
                            block, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f'Expected one {node} attachment in craft part')
    return result


def make_transfer_craft(source: str) -> str:
    """Add two stock FL-T200 snapshots to OrbiterRL Terrier's upper stack.

    This preserves the crew pod and the sequential Swivel/Terrier stage plan.
    Two 125 kg dry / 1000 kg propellant tanks extend the upper stage without
    editing stock assets or changing the existing lower-stage resource groups.
    """
    prefix, blocks = _split_parts(source)
    if not re.search(r'^ship = OrbiterRL Terrier$', prefix, re.MULTILINE) or len(blocks) != 18:
        raise ValueError('Expected the 18-part OrbiterRL Terrier source craft')
    parts = {_part_id(block): block for block in blocks}
    if len(parts) != len(blocks):
        raise ValueError('Source craft contains duplicate part IDs')
    engines = [identifier for identifier in parts if identifier.startswith('liquidEngine3.v2_')]
    if len(engines) != 1:
        raise ValueError('Expected exactly one upper Terrier engine')
    engine_id = engines[0]
    engine = parts[engine_id]
    tank_id = _attachment_peer(engine, 'top')
    if tank_id not in parts or not tank_id.startswith('fuelTankSmall_'):
        raise ValueError('The upper Terrier must attach below an FL-T200 tank')
    tank = parts[tank_id]
    if re.findall(r'^\tlink = (\S+)', tank, re.MULTILINE) != [engine_id]:
        raise ValueError('The cloned upper tank must have only the Terrier child')
    top, bottom = _attachment_vector(tank, 'top'), _attachment_vector(tank, 'bottom')
    if any(abs(v) > 1e-5 for v in (*top[::2], *bottom[::2])) or top[1] <= bottom[1]:
        raise ValueError('Only coaxial vertical tank stacks are supported')
    resources = dict(re.findall(r'\tRESOURCE\n\t\{\n\t\tname = (\S+)\n\t\tamount = ([^\n]+)', tank))
    if resources != {'LiquidFuel': '90', 'Oxidizer': '110'}:
        raise ValueError('Expected a full stock FL-T200 with 90 LiquidFuel and 110 Oxidizer')
    persistent_ids = re.findall(r'^\tpersistentId = (\d+)$', source, re.MULTILINE)
    if len(persistent_ids) != len(blocks) or len(set(persistent_ids)) != len(blocks):
        raise ValueError('Every source part needs a unique persistentId')
    used = {int(identifier.rsplit('_', 1)[1]) for identifier in parts}
    used.update(map(int, persistent_ids))
    new_numbers, candidate = [], (max(used)+1) % (2**32)
    while len(new_numbers) < 4:
        if candidate != 0 and candidate not in used:
            new_numbers.append(candidate)
            used.add(candidate)
        candidate = (candidate+1) % (2**32)
    new_ids = ['fuelTankSmall_'+str(value) for value in new_numbers[:2]]
    height, shift = top[1]-bottom[1], -2*(top[1]-bottom[1])
    descendants, pending = set(), [engine_id]
    while pending:
        identifier = pending.pop()
        if identifier in descendants or identifier not in parts:
            raise ValueError('Invalid lower craft attachment tree')
        descendants.add(identifier)
        pending.extend(re.findall(r'^\tlink = (\S+)', parts[identifier], re.MULTILINE))
    clones = []
    for index, new_id in enumerate(new_ids):
        clone = tank.replace(f'\tpart = {tank_id}\n', f'\tpart = {new_id}\n')
        clone = re.sub(r'^\tpersistentId = \d+$', f'\tpersistentId = {new_numbers[index+2]}',
                       clone, flags=re.MULTILINE)
        position = list(_vector(tank, 'pos'))
        position[1] -= height*(index+1)
        clone = _set_vector(clone, 'pos', position)
        clone = _set_vector(clone, 'attPos0', (0., -height, 0.))
        parent = tank_id if index == 0 else new_ids[index-1]
        child = new_ids[index+1] if index+1 < len(new_ids) else engine_id
        clone = _set_attachment_peer(clone, 'top', parent)
        clone = _set_attachment_peer(clone, 'bottom', child)
        clone = clone.replace(f'\tlink = {engine_id}\n', f'\tlink = {child}\n')
        clones.append(clone)
    rebuilt = []
    for block in blocks:
        identifier = _part_id(block)
        if identifier == tank_id:
            block = block.replace(f'\tlink = {engine_id}\n', f'\tlink = {new_ids[0]}\n')
            block = _set_attachment_peer(block, 'bottom', new_ids[0])
        if identifier in descendants:
            position = list(_vector(block, 'pos'))
            position[1] += shift
            block = _set_vector(block, 'pos', position)
            if identifier == engine_id:
                block = _set_attachment_peer(block, 'top', new_ids[-1])
                parent_position = _vector(clones[-1], 'pos')
                block = _set_vector(block, 'attPos0', tuple(a-b for a, b in zip(position, parent_position)))
        rebuilt.append(block)
        if identifier == tank_id:
            rebuilt.extend(clones)
    prefix = re.sub(r'^ship = .*$', 'ship = OrbiterRL Transfer', prefix, flags=re.MULTILINE)
    prefix = re.sub(r'^description = .*$',
                    'description = Crewed Swivel/Terrier transfer craft with four upper FL-T200 tanks; sequential stages 3 and 2.',
                    prefix, flags=re.MULTILINE)
    size = re.search(r'^size = ([^\n]+)$', prefix, re.MULTILINE)
    if size:
        dimensions = [float(value) for value in size.group(1).split(',')]
        dimensions[1] -= shift
        prefix = prefix[:size.start(1)]+','.join(f'{v:.10g}' for v in dimensions)+prefix[size.end(1):]
    return prefix + ''.join(rebuilt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument('--terrier-template', type=Path,
                         help='Stock craft containing exactly one Terrier (for example Science Jr.craft)')
    variant.add_argument('--transfer', action='store_true',
                         help='Source is OrbiterRL Terrier; insert two additional upper FL-T200 tanks')
    args = parser.parse_args()
    # KSP's legacy stock craft uses a non-UTF8 description separator.
    source = args.source.read_text(errors='replace')
    result = (make_transfer_craft(source) if args.transfer else
              make_terrier_craft(source, args.terrier_template.read_text(errors='replace'))
              if args.terrier_template else make_liquid_craft(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(result)
    print(f'Created {args.output}: {20 if args.transfer else 18} parts; activation stages 3 (launch), 2 (separate/ignite).')


if __name__ == '__main__':
    main()
