def gray_encode(n: int) -> int:
    return n ^ (n >> 1)


def gray_decode(g: int) -> int:
    n = g
    shift = 1
    while (g >> shift) > 0:
        n ^= (g >> shift)
        shift += 1
    return n