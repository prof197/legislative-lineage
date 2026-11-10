import shutil
import ssl
import zipfile
from pathlib import Path

import pytest

from lineage.disk import InsufficientDiskSpaceError
from lineage.fetch import USER_AGENT, fetch_raw, moved, request_url, tls_context
from lineage.snapshot import load_manifest, scan, verify, write_manifest
from tests.conftest import Upstream


@pytest.fixture
def manifest(raw_dir: Path, tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    write_manifest(scan(raw_dir), path)
    return path


GA = "GA/GA_2025_26_json_fixture.zip"


def test_downloads_every_archive_and_identifies_the_client(
    upstream: Upstream, manifest: Path, tmp_path: Path
) -> None:
    raw = tmp_path / "fetched"

    result = fetch_raw(raw, manifest, source=upstream.url, floor_bytes=0, pause=0)

    assert (result.fetched, result.present, result.problems) == (4, 0, [])
    verify(raw, load_manifest(manifest))
    assert not list(raw.glob("*/*.part"))
    assert upstream.user_agents == [USER_AGENT] * 4


def test_skips_archives_already_present(
    upstream: Upstream, manifest: Path, raw_dir: Path, tmp_path: Path
) -> None:
    raw = tmp_path / "fetched"
    (raw / "GA").mkdir(parents=True)
    shutil.copy(raw_dir / GA, raw / GA)

    result = fetch_raw(raw, manifest, source=upstream.url, floor_bytes=0, pause=0)

    assert (result.fetched, result.present) == (3, 1)
    assert len(upstream.user_agents) == 3


def test_refuses_a_changed_archive_by_default(
    upstream: Upstream, manifest: Path, tmp_path: Path
) -> None:
    served = upstream.file(Path(GA).name)
    data = bytearray(served.read_bytes())
    data[-1] ^= 0xFF  # same size, different bytes
    served.write_bytes(bytes(data))
    pinned_before = manifest.read_text()
    raw = tmp_path / "fetched"

    result = fetch_raw(raw, manifest, source=upstream.url, floor_bytes=0, pause=0)

    assert result.fetched == 3
    assert result.problems == [f"{GA}: SHA-256 differs from the manifest"]
    assert not (raw / GA).exists()
    assert not list(raw.glob("*/*.part"))
    assert manifest.read_text() == pinned_before


def test_refuses_a_resized_archive_before_downloading_it(
    upstream: Upstream, manifest: Path, tmp_path: Path
) -> None:
    with upstream.file(Path(GA).name).open("ab") as f:
        f.write(b"\0")
    raw = tmp_path / "fetched"

    result = fetch_raw(raw, manifest, source=upstream.url, floor_bytes=0, pause=0)

    assert len(result.problems) == 1
    assert result.problems[0].startswith(f"{GA}: server offers")
    assert not (raw / GA).exists()


def test_repin_accepts_a_regenerated_archive_and_records_it(
    upstream: Upstream, manifest: Path, tmp_path: Path
) -> None:
    served = upstream.file(Path(GA).name)
    with zipfile.ZipFile(served) as z:
        members = {n: z.read(n) for n in z.namelist()}
    members["README"] = members["README"].replace(b"2026-07-25", b"2026-10-01")
    with zipfile.ZipFile(served, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    raw = tmp_path / "fetched"

    result = fetch_raw(raw, manifest, source=upstream.url, floor_bytes=0, pause=0, repin=True)

    assert (result.fetched, result.repinned, result.problems) == (4, [GA], [])
    repinned = {a.path: a for a in load_manifest(manifest)}
    assert repinned[GA].generated_at.startswith("2026-10-01")
    verify(raw, list(repinned.values()))


def test_reports_an_archive_the_server_no_longer_has(
    upstream: Upstream, manifest: Path, tmp_path: Path
) -> None:
    upstream.file(Path(GA).name).unlink()
    raw = tmp_path / "fetched"

    result = fetch_raw(raw, manifest, source=upstream.url, floor_bytes=0, pause=0)

    assert result.fetched == 3
    assert len(result.problems) == 1
    assert result.problems[0].startswith(f"{GA}: HTTP Error 404")


def test_refuses_to_start_below_the_disk_floor(
    upstream: Upstream, manifest: Path, tmp_path: Path
) -> None:
    with pytest.raises(InsufficientDiskSpaceError):
        fetch_raw(tmp_path / "fetched", manifest, source=upstream.url, floor_bytes=1 << 60, pause=0)

    assert upstream.user_agents == []


def test_the_intermediate_bundle_adds_certificates_without_relaxing_verification() -> None:
    context = tls_context()

    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    added = (
        context.cert_store_stats()["x509_ca"]
        - ssl.create_default_context().cert_store_stats()["x509_ca"]
    )
    assert added == 3


def test_illinois_beta_links_are_fetched_from_the_main_site() -> None:
    url = "https://beta.ilga.gov/Legislation/BillStatus/FullText?LegDocId=1&Print=1"

    assert moved(url) == "https://ilga.gov/Legislation/BillStatus/FullText?LegDocId=1&Print=1"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "https://search-prod.lis.state.oh.us/solarapi/v1/general_assembly_135/bills/sb237/EN/05/"
            "sb237_05_EN?format=pdf",
            "https://search-prod.lis.state.oh.us/api/v2/general_assembly_135/legislation/sb237/"
            "05_EN/pdf/",
        ),
        (
            "https://search-prod.lis.state.oh.us/solarapi/v1/general_assembly_135_special_1/bills/"
            "hb2/EN/04/hb2_04_EN?format=pdf",
            "https://search-prod.lis.state.oh.us/api/v2/general_assembly_135_special_1/legislation/"
            "hb2/04_EN/pdf/",
        ),
    ],
)
def test_ohio_links_in_its_retired_scheme_are_fetched_from_the_current_one(
    old: str, new: str
) -> None:
    assert moved(old) == new
    # The current scheme, and anything the rule doesn't describe, is left as it is.
    assert moved(new) == new
    html = old.replace("format=pdf", "format=html")
    assert moved(html) == html


def test_spaces_in_a_link_are_escaped_and_existing_escapes_kept() -> None:
    url = "https://www.oklegislature.gov/cf_pdf/2025-26 ENR/SB/SB1232 ENR.PDF?a=1&b=%20"

    assert request_url(url) == (
        "https://www.oklegislature.gov/cf_pdf/2025-26%20ENR/SB/SB1232%20ENR.PDF?a=1&b=%20"
    )
