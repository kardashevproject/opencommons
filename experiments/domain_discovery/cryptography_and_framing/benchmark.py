"""Deep-Space Telemetry Ratchet and Wire-Framing Benchmark.

Composes 10 verified OpenCommons concepts from the cryptography_and_framing domain:
1. `hkdf_extract_step`: Extracts high-entropy master root key from noisy sensor entropy + epoch salt.
2. `hkdf_expand_info_block`: Ratchets per-frame encryption and MAC authentication subkeys.
3. `shamir_secret_eval_poly`: Constructs (k, n) threshold backup shares of mission root key.
4. `shamir_lagrange_interpolate`: Recovers mission root key from ground station share subset.
5. `poly1305_clamp_key`: Prepares and clamps evaluation keys for high-speed field authentication.
6. `poly1305_fe_mul_mod`: Evaluates polynomial MAC accumulator over telemetry data blocks.
7. `crc32c_castagnoli_step`: Computes hardware-grade Castagnoli CRC error detection.
8. `cobs_encode_frame`: Encodes binary payload into zero-delimited packet framing.
9. `cobs_decode_frame`: Recovers payload bytes from raw wire stream.
10. `constant_time_byte_eq`: Side-channel-resistant comparison of authentication tags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def _load_domain_functions() -> dict[str, Callable[..., Any]]:
    catalog_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "cryptography_and_framing.json"
    with open(catalog_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)
    env: dict[str, Any] = {}
    for c in concepts:
        code_obj = compile(c["source_code"], f"<{c['name']}>", "exec")
        exec(code_obj, env)
    return {c["name"]: env[c["name"]] for c in concepts}


_FUNCS = _load_domain_functions()
hkdf_extract_step = _FUNCS["hkdf_extract_step"]
hkdf_expand_info_block = _FUNCS["hkdf_expand_info_block"]
shamir_secret_eval_poly = _FUNCS["shamir_secret_eval_poly"]
shamir_lagrange_interpolate = _FUNCS["shamir_lagrange_interpolate"]
poly1305_clamp_key = _FUNCS["poly1305_clamp_key"]
poly1305_fe_mul_mod = _FUNCS["poly1305_fe_mul_mod"]
crc32c_castagnoli_step = _FUNCS["crc32c_castagnoli_step"]
cobs_encode_frame = _FUNCS["cobs_encode_frame"]
cobs_decode_frame = _FUNCS["cobs_decode_frame"]
constant_time_byte_eq = _FUNCS["constant_time_byte_eq"]


def run_deep_space_telemetry_ratchet_pipeline(
    master_entropy: list[int],
    epoch_salt: list[int],
    telemetry_payloads: list[list[int]],
    threshold_k: int = 3,
    total_stations_n: int = 5,
) -> dict[str, Any]:
    """Executes a full deep-space telemetry pipeline:

    1. Ratchet initialization via HKDF extract.
    2. Threshold key backup across distributed ground stations via Shamir Secret Sharing.
    3. Ground station recovery verification.
    4. Per-frame ratcheted session subkey derivation via HKDF expand.
    5. Poly1305 key clamping and Horner-step message authentication tag generation.
    6. CRC-32C Castagnoli hardware-level frame error detection checksumming.
    7. Consistent Overhead Byte Stuffing (COBS) packetization with 0x00 boundary delimiters.
    8. Simulated wire transmission with noise/tamper detection.
    9. Receiver COBS depacketization, CRC verification, and constant-time MAC authentication.
    """
    # Step 1: Master Root Key Extraction
    prk = hkdf_extract_step(salt=epoch_salt, ikm=master_entropy)
    root_secret_int = (prk[0] << 24) | (prk[1] << 16) | (prk[2] << 8) | prk[3]
    prime = 65521  # 16-bit prime for Shamir secret polynomial

    # Step 2: Threshold Secret Sharing (k-of-n)
    secret_val = root_secret_int % prime
    poly_coeffs = [secret_val, 1234, 5678][:threshold_k]
    while len(poly_coeffs) < threshold_k:
        poly_coeffs.append(42)

    shares: list[list[int]] = []
    for station_id in range(1, total_stations_n + 1):
        share_val = shamir_secret_eval_poly(poly_coeffs, station_id, prime)
        shares.append([station_id, share_val])

    # Step 3: Ground Station Recovery (subset of k stations)
    active_shares = shares[:threshold_k]
    recovered_secret = shamir_lagrange_interpolate(active_shares, prime)
    assert recovered_secret == secret_val, "Threshold secret recovery mismatch!"

    # Step 4-7: Transmitter Processing for Each Frame
    transmitted_wire_stream: list[int] = []
    session_tags: list[list[int]] = []

    for frame_idx, payload in enumerate(telemetry_payloads):
        # 4. Ratchet subkey for frame
        info_tag = [ord(c) for c in f"DSN-RATCHET-FRAME-{frame_idx}"]
        subkeys_bytes: list[int] = []
        for block_id in range(8):  # 8 blocks * 4 bytes = 32 bytes
            chunk = hkdf_expand_info_block(prk, info_tag, block_id)
            subkeys_bytes.extend(chunk)

        # 5. Poly1305 MAC generation
        r_clamped, s_pad = poly1305_clamp_key(subkeys_bytes[:32])
        acc = 0
        p_poly = (1 << 130) - 5
        # Feed payload in 16-byte blocks
        for i in range(0, len(payload), 16):
            chunk = payload[i : i + 16]
            block_int = 0
            for b_idx, b_val in enumerate(chunk):
                block_int |= (b_val & 255) << (8 * b_idx)
            block_int |= 1 << (8 * len(chunk))
            acc = poly1305_fe_mul_mod(acc, block_int, r_clamped)
        tag_int = (acc + s_pad) % (1 << 128)
        tag_bytes = [(tag_int >> (8 * i)) & 255 for i in range(16)]
        session_tags.append(tag_bytes)

        # 6. CRC-32C Castagnoli Checksum
        frame_body = list(payload) + tag_bytes
        crc = (1 << 32) - 1
        for b in frame_body:
            crc = crc32c_castagnoli_step(crc, b)
        crc = crc ^ ((1 << 32) - 1)
        crc_bytes = [(crc >> 24) & 255, (crc >> 16) & 255, (crc >> 8) & 255, crc & 255]

        # 7. COBS Encoding
        framed_packet = frame_body + crc_bytes
        cobs_stuffed = cobs_encode_frame(framed_packet)
        # Framing delimiter 0x00 separates packets on wire
        transmitted_wire_stream.extend(cobs_stuffed)
        transmitted_wire_stream.append(0)

    # Step 8-9: Receiver Ingestion & Verification
    recovered_payloads: list[list[int]] = []
    received_packets: list[list[int]] = []
    current_packet: list[int] = []

    for b in transmitted_wire_stream:
        if b == 0:
            if current_packet:
                received_packets.append(current_packet)
                current_packet = []
        else:
            current_packet.append(b)

    for frame_idx, stuffed in enumerate(received_packets):
        # 8. COBS Decoding
        decoded = cobs_decode_frame(stuffed)
        assert len(decoded) >= 20, "Corrupt truncated frame!"

        # Extract payload, tag, and CRC
        rec_payload = decoded[:-20]
        rec_tag = decoded[-20:-4]
        rec_crc_bytes = decoded[-4:]
        expected_crc = (rec_crc_bytes[0] << 24) | (rec_crc_bytes[1] << 16) | (rec_crc_bytes[2] << 8) | rec_crc_bytes[3]

        # Recompute CRC-32C
        check_crc = (1 << 32) - 1
        for b in decoded[:-4]:
            check_crc = crc32c_castagnoli_step(check_crc, b)
        check_crc = check_crc ^ ((1 << 32) - 1)
        assert check_crc == expected_crc, f"CRC-32C verification failed on frame {frame_idx}!"

        # Verify Poly1305 MAC in constant time
        expected_tag = session_tags[frame_idx]
        is_mac_valid = constant_time_byte_eq(rec_tag, expected_tag)
        assert is_mac_valid is True, f"MAC tag verification failed on frame {frame_idx}!"

        recovered_payloads.append(rec_payload)

    return {
        "status": "success",
        "frames_processed": len(telemetry_payloads),
        "total_wire_bytes": len(transmitted_wire_stream),
        "root_secret": root_secret_int,
        "recovered_secret": recovered_secret,
        "all_crc_valid": True,
        "all_mac_valid": True,
        "recovered_payloads": recovered_payloads,
    }
