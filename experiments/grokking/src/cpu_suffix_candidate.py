"""Experimental suffix execution only; not used by any training run."""
from torch.nn import functional as F


def finish(model, x, first):
    """Keep full causal context through last attention, then tokenwise RHS only."""
    for l in range(first, 2):
        block = model.ref.decoder.blocks[l]
        q, k, v = [model._qkv(block, kind, x) for kind in 'qkv']
        hub = model.attention(q, k, v)
        if l == 1:
            x, hub = x[:, -2:], hub[:, -2:]
        writer = block.self_attn.Wo(hub)
        x = model._norm(x+writer, block.self_attn_norm, f'b{l}.norm1', False)
        ff = block.ffn.ffn[2](block.ffn.ffn[0](x).relu())
        x = model._norm(x+ff, block.ffn_norm, f'b{l}.norm2', False)
    return F.pad(model.ref.linear(x), (0, 0, 4, 0))


def terminal_suffix(model, site, noise):
    b = model.outputs[site].shape[0]; n = noise.shape[0]//b
    terminal = site.startswith('b1.')
    def repeat(value):
        return (value[:, -2:] if terminal else value).repeat(n, 1, 1)
    x = repeat(model.outputs[site])+(noise[:, -2:] if terminal else noise)
    if site == 'embedding':
        return finish(model, x+model.ref.position_encoding[:x.shape[1]], 0)
    l = int(site[1]); kind = site.split('.')[1]
    block = model.ref.decoder.blocks[l]; s = f'b{l}'
    residual = repeat(model.outputs[s+'.norm1'])
    if kind == 'hub':
        x = block.self_attn.Wo(x); kind = 'writer'
    if kind == 'writer':
        x = model._norm(repeat(model.cache[s+'.x'])+x, block.self_attn_norm, s+'.norm1', False)
        kind = 'norm1'
    if kind == 'norm1':
        residual = x; x = block.ffn.ffn[0](x); kind = 'hidden'
    if kind == 'hidden':
        x = block.ffn.ffn[2](x.relu()); kind = 'ffout'
    if kind == 'ffout':
        x = model._norm(residual+x, block.ffn_norm, s+'.norm2', False)
    return finish(model, x, l+1)
