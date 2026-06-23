import socket
import struct
import json
import numpy as np

HEADER_FMT = "!IHH"          # frame_id, chunk_idx, total_chunks
HEADER_SIZE = struct.calcsize(HEADER_FMT)
CHUNK_BYTES = 1024 - HEADER_SIZE


class UDPFrameSender:
    """Отправляет произвольный bytes-payload, разбивая на чанки."""
    def __init__(self, dst_ip: str, dst_port: int):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dst = (dst_ip, dst_port)

    def send(self, frame_id: int, payload: bytes):
        total_chunks = max(1, (len(payload) + CHUNK_BYTES - 1) // CHUNK_BYTES)
        for idx in range(total_chunks):
            chunk = payload[idx * CHUNK_BYTES:(idx + 1) * CHUNK_BYTES]
            header = struct.pack(HEADER_FMT, frame_id, idx, total_chunks)
            self.sock.sendto(header + chunk, self.dst)

    def close(self):
        self.sock.close()


class UDPFrameReceiver:
    def __init__(self, listen_ip: str, listen_port: int, timeout: float = 10.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((listen_ip, listen_port))
        self.sock.settimeout(timeout)
        self._buffers = {}

    def recv(self) -> bytes | None:
        while True:
            try:
                packet, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                return None

            frame_id, chunk_idx, total_chunks = struct.unpack(HEADER_FMT, packet[:HEADER_SIZE])
            payload = packet[HEADER_SIZE:]

            buf = self._buffers.setdefault(frame_id, {})
            buf[chunk_idx] = payload

            if len(buf) == total_chunks:
                ordered = b"".join(buf[i] for i in range(total_chunks))
                del self._buffers[frame_id]
                return ordered

    def close(self):
        self.sock.close()


# ── Упаковка кадра: метаданные + сигнал ────────────────────────────────────

def pack_frame(snr_db, n_bits, tx_bits: np.ndarray, rx_signal: np.ndarray,
               snr_idx, total_snrs, iter_idx, total_iters) -> bytes:
    meta = {
        "snr_db": snr_db,
        "n_bits": n_bits,
        "snr_idx": snr_idx,
        "total_snrs": total_snrs,
        "iter_idx": iter_idx,
        "total_iters": total_iters,
    }
    meta_bytes = json.dumps(meta).encode("utf-8")
    tx_bits_bytes = np.packbits(tx_bits).tobytes()
    sig_bytes = rx_signal.astype(np.complex64).tobytes()

    return b"".join([
        struct.pack("!I", len(meta_bytes)), meta_bytes,
        struct.pack("!I", len(tx_bits_bytes)), tx_bits_bytes,
        struct.pack("!I", len(sig_bytes)), sig_bytes,
    ])


def unpack_frame(payload: bytes):
    off = 0
    (meta_len,) = struct.unpack_from("!I", payload, off); off += 4
    meta = json.loads(payload[off:off + meta_len]); off += meta_len

    (bits_len,) = struct.unpack_from("!I", payload, off); off += 4
    tx_bits_packed = payload[off:off + bits_len]; off += bits_len

    (sig_len,) = struct.unpack_from("!I", payload, off); off += 4
    sig_bytes = payload[off:off + sig_len]

    n_bits = meta["n_bits"]
    tx_bits = np.unpackbits(np.frombuffer(tx_bits_packed, dtype=np.uint8))[:n_bits]
    rx_signal = np.frombuffer(sig_bytes, dtype=np.complex64)

    return meta, tx_bits, rx_signal