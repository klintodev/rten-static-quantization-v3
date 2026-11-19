"""Small ONNX and SafeTensors fixtures using the public wire formats."""

import json
import math
import struct


def varint(n):
    n %= 2**64
    parts = []
    while n > 127:
        parts.append(128 + n % 128)
        n //= 128
    return bytes(parts + [n])


def field(tag, value):
    if isinstance(value, int):
        return varint(tag << 3) + varint(value)
    if isinstance(value, str):
        value = value.encode()
    return varint(tag << 3 | 2) + varint(len(value)) + value


def fields(tag, items):
    return b"".join(field(tag, item) for item in items)


def unpack(blob):
    def read(cursor):
        val = 0
        for shift in range(0, 70, 7):
            b = blob[cursor]
            cursor += 1
            val += (b & 127) << shift
            if b < 128:
                return val, cursor
        raise ValueError("invalid varint")

    result = {}
    i = 0
    while i < len(blob):
        key, i = read(i)
        wire = key % 8
        if wire == 0:
            value, i = read(i)
        else:
            if wire == 2:
                n, i = read(i)
            else:
                n = {1: 8, 5: 4}[wire]
            value, i = blob[i : i + n], i + n
        result.setdefault(key >> 3, []).append(value)
    return result


DT = {
    1: ("F32", "f"),
    2: ("U8", "B"),
    3: ("I8", "b"),
    6: ("I32", "i"),
    7: ("I64", "q"),
    9: ("BOOL", "?"),
}


def data(dtype, vals):
    return struct.pack("<" + str(len(vals)) + DT[dtype][1], *vals)


def tensor(name, dtype, shape, vals=None, external=None):
    body = field(8, name) + field(2, dtype) + fields(1, shape)
    if external:
        path, off, length = external
        for k, v in [("location", path), ("offset", str(off)), ("length", str(length))]:
            body += field(13, field(1, k) + field(2, v))
        return body + field(14, 1)
    return body + field(9, data(dtype, vals))


def vi(name, dtype, shape):
    dims = fields(1, [field(2, d) if isinstance(d, str) else field(1, d) for d in shape])
    return field(1, name) + field(2, field(1, field(1, dtype) + field(2, dims)))


def op(kind, ins, outs, **attrs):
    result = field(4, kind) + fields(1, ins) + fields(2, outs)
    for k, v in attrs.items():
        if isinstance(v, bytes):
            payload = field(20, 5) + field(6, v)
        elif isinstance(v, list):
            payload = field(20, 7) + fields(8, v)
        else:
            payload = field(20, 2) + field(3, v)
        result += field(5, field(1, k) + payload)
    return result


def graph(nodes, inputs, outputs, weights=(), name="main"):
    return (
        fields(1, nodes)
        + field(2, name)
        + fields(5, weights)
        + fields(11, inputs)
        + fields(12, outputs)
    )


def model(path, g):
    path.write_bytes(field(1, 8) + field(7, g) + field(8, field(1, "") + field(2, 18)))
    return path


def safetensors(path, **named):
    header = {}
    raw = bytearray()
    for k, (dt, shape, vals) in named.items():
        start = len(raw)
        assert math.prod(shape) == len(vals)
        raw.extend(data(dt, vals))
        header[k] = dict(dtype=DT[dt][0], shape=list(shape), data_offsets=[start, len(raw)])
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + raw)
    return path


def inspect(path):
    return unpack(unpack(path.read_bytes())[7][0])


def walk(g):
    for item in g.get(1, []):
        n = unpack(item)
        yield n
        for attr in n.get(5, []):
            for sub in unpack(attr).get(6, []):
                yield from walk(unpack(sub))


def ops(g):
    return [n[4][0].decode() for n in walk(g)]


def branches(g, kind):
    n = next(n for n in walk(g) if n[4] == [kind.encode()])
    return {a[1][0].decode(): unpack(a[6][0]) for a in map(unpack, n.get(5, [])) if 6 in a}


def interface(g, fieldnum):
    out = []
    for v in map(unpack, g.get(fieldnum, [])):
        vtype = unpack(unpack(v[2][0])[1][0])
        dims = [unpack(d) for d in unpack(vtype[2][0]).get(1, [])]
        out.append((v[1][0], vtype[1][0], [(d.get(1) or d.get(2) or [None])[0] for d in dims]))
    return out


def members(path):
    result = {path.resolve()}

    def visit(g):
        for tensor in map(unpack, g.get(5, [])):
            ext = {e[1][0]: e[2][0] for e in map(unpack, tensor.get(13, []))}
            if b"location" in ext:
                member = (path.parent / ext[b"location"].decode()).resolve()
                assert member.is_relative_to(path.parent.resolve()), member
                result.add(member)
        for n in map(unpack, g.get(1, [])):
            for a in map(unpack, n.get(5, [])):
                for s in a.get(6, []):
                    visit(unpack(s))

    visit(inspect(path))
    return result
