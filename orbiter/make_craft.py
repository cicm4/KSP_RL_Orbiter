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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--terrier-template', type=Path,
                        help='Stock craft containing exactly one Terrier (for example Science Jr.craft)')
    args = parser.parse_args()
    # KSP's legacy stock craft uses a non-UTF8 description separator.
    source = args.source.read_text(errors='replace')
    result = (make_terrier_craft(source, args.terrier_template.read_text(errors='replace'))
              if args.terrier_template else make_liquid_craft(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(result)
    print(f'Created {args.output}: 18 parts; activation stages 3 (launch), 2 (separate/ignite).')


if __name__ == '__main__':
    main()
