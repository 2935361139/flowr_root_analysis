"""Model-independent graph regions. Region membership is NOT a causal label."""
from collections import deque


def atom_regions(atom_ids, edges, fixed_ids, reference_anchor_ids):
    ids, fixed = set(atom_ids), set(fixed_ids)
    if not fixed <= ids or not set(reference_anchor_ids) <= fixed:
        raise ValueError('Fixed / reference anchor mapping is outside the graph')
    neighbors = {i: set() for i in ids}
    boundary = set()
    for a, b in edges:
        if a not in ids or b not in ids:
            raise ValueError('Edge endpoint outside graph')
        neighbors[a].add(b); neighbors[b].add(a)
        if (a in fixed) != (b in fixed):
            boundary.update((a, b))
    distance = {i: 0 for i in boundary}
    queue = deque(sorted(boundary))
    while queue:
        i = queue.popleft()
        for j in neighbors[i]:
            if j not in distance:
                distance[j] = distance[i] + 1
                queue.append(j)
    return {i: {
        'region': ('fixed_boundary' if i in boundary else 'fixed_other') if i in fixed
                  else ('generated_boundary' if i in boundary else 'generated_other'),
        'is_reference_attachment_atom': i in reference_anchor_ids,
        'graph_hops_to_current_boundary': distance.get(i),
    } for i in ids}


def fixed_pair_comparison(atom_id, environment_id, fixed_to_reference, reference_pairs):
    if atom_id not in fixed_to_reference:
        return 'no_reference_atom_correspondence_for_generated_core'
    return ('same_fixed_atom_environment_pair_already_clashes_in_reference'
            if (fixed_to_reference[atom_id], environment_id) in reference_pairs
            else 'fixed_atom_environment_pair_does_not_clash_in_reference')
