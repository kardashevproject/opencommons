from pathlib import Path
import sys
import pytest

repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from experiments.domain_discovery.cryptography_and_framing.benchmark import (
    run_deep_space_telemetry_ratchet_pipeline,
    hkdf_extract_step,
    hkdf_expand_info_block,
    shamir_secret_eval_poly,
    shamir_lagrange_interpolate,
    poly1305_clamp_key,
    poly1305_fe_mul_mod,
    crc32c_castagnoli_step,
    cobs_encode_frame,
    cobs_decode_frame,
    constant_time_byte_eq,
)


@pytest.fixture
def sample_telemetry_dataset():
    master_entropy = [0x5A, 0x1F, 0xC3, 0x88, 0x90, 0x4D, 0x22, 0xEF]
    epoch_salt = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
    # 4 distinct telemetry frames (star tracker, IMU gyros, battery voltage, temperature sensors)
    payloads = [
        [10, 20, 30, 40, 50, 60, 70, 80],
        [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 255, 254],
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
        [42] * 32,
    ]
    return master_entropy, epoch_salt, payloads


def test_deep_space_telemetry_pipeline_execution(sample_telemetry_dataset):
    master_entropy, epoch_salt, payloads = sample_telemetry_dataset
    res = run_deep_space_telemetry_ratchet_pipeline(
        master_entropy=master_entropy,
        epoch_salt=epoch_salt,
        telemetry_payloads=payloads,
        threshold_k=3,
        total_stations_n=5,
    )
    assert res["status"] == "success"
    assert res["frames_processed"] == len(payloads)
    assert res["all_crc_valid"] is True
    assert res["all_mac_valid"] is True
    assert res["recovered_payloads"] == payloads


def test_shamir_threshold_recovery_variations():
    secret = 31415
    prime = 65521
    k = 3
    poly = [secret, 999, 888]
    # Generate 5 shares
    shares = [[x, shamir_secret_eval_poly(poly, x, prime)] for x in range(1, 6)]

    # Any subset of 3 shares must recover secret exactly
    for subset in [
        [shares[0], shares[1], shares[2]],
        [shares[0], shares[2], shares[4]],
        [shares[1], shares[3], shares[4]],
        [shares[2], shares[3], shares[4]],
    ]:
        rec = shamir_lagrange_interpolate(subset, prime)
        assert rec == secret


def test_cobs_framing_guarantee():
    # Any payload with multiple zeros must encode into zero-free stream
    dirty_payload = [0, 0, 1, 0, 2, 3, 0, 0, 4, 0]
    encoded = cobs_encode_frame(dirty_payload)
    assert 0 not in encoded, "COBS encoded buffer must not contain delimiter zero!"
    decoded = cobs_decode_frame(encoded)
    assert decoded == dirty_payload


def test_crc32c_castagnoli_error_detection():
    data = [1, 2, 3, 4, 5, 6, 7, 8]
    crc_orig = (1 << 32) - 1
    for b in data:
        crc_orig = crc32c_castagnoli_step(crc_orig, b)
    crc_orig ^= (1 << 32) - 1

    # Tampered single bit
    tampered_data = list(data)
    tampered_data[4] ^= 0x01
    crc_tampered = (1 << 32) - 1
    for b in tampered_data:
        crc_tampered = crc32c_castagnoli_step(crc_tampered, b)
    crc_tampered ^= (1 << 32) - 1

    assert crc_orig != crc_tampered


def test_constant_time_byte_eq_correctness():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    c = [1, 2, 3, 4, 6]
    d = [1, 2, 3, 4]

    assert constant_time_byte_eq(a, b) is True
    assert constant_time_byte_eq(a, c) is False
    assert constant_time_byte_eq(a, d) is False


def test_ratchet_forward_secrecy():
    prk = [10, 20, 30, 40]
    info = [ord(c) for c in "TELEMETRY-STREAM"]
    block0 = hkdf_expand_info_block(prk, info, 0)
    block1 = hkdf_expand_info_block(prk, info, 1)
    block2 = hkdf_expand_info_block(prk, info, 2)

    assert block0 != block1
    assert block1 != block2
    assert block0 != block2


def test_composed_concepts_registered_in_catalog():
    """Verify that all composed building blocks exist in cryptography_and_framing.json."""
    cat_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "domains" / "cryptography_and_framing.json"
    assert cat_path.is_file()
    import json
    with open(cat_path, "r", encoding="utf-8") as f:
        concepts = json.load(f)

    catalog_names = {c["name"] for c in concepts}
    required = {
        "hkdf_extract_step",
        "hkdf_expand_info_block",
        "shamir_secret_eval_poly",
        "shamir_lagrange_interpolate",
        "poly1305_clamp_key",
        "poly1305_fe_mul_mod",
        "crc32c_castagnoli_step",
        "cobs_encode_frame",
        "cobs_decode_frame",
        "constant_time_byte_eq",
    }
    assert required.issubset(catalog_names), f"Missing concepts: {required - catalog_names}"

