import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from fovea.core.export.xmp import Sidecar, render, sidecar_path, write_sidecar

RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
XMP = "{http://ns.adobe.com/xap/1.0/}"


def parse_description(xmp_text: str) -> ET.Element:
    body = xmp_text.split("?>", 1)[1].rsplit("<?xpacket", 1)[0]
    return next(iter(ET.fromstring(body).iter(f"{RDF}Description")))


def test_sidecar_path_replaces_extension() -> None:
    assert sidecar_path(Path("/x/IMG_1234.CR3")) == Path("/x/IMG_1234.xmp")


def test_render_is_valid_xml_with_expected_fields() -> None:
    s = Sidecar(
        rating=4,
        label="Green",
        keywords=["Fovea|Tier A", "Nature|Birds|Anatidae|Mallard"],
        fovea={"FocusScore": 87, "BlurType": "defocus"},
    )
    desc = parse_description(render(s))
    assert desc.get(f"{XMP}Rating") == "4"
    assert desc.get(f"{XMP}Label") == "Green"
    assert desc.get("{https://github.com/JamesZhang22/fovea/xmp/1.0/}FocusScore") == "87"
    flat = [
        li.text for li in desc.find("{http://purl.org/dc/elements/1.1/}subject").iter(f"{RDF}li")
    ]
    assert flat == ["Tier A", "Mallard"]
    hier = [
        li.text
        for li in desc.find("{http://ns.adobe.com/lightroom/1.0/}hierarchicalSubject").iter(
            f"{RDF}li"
        )
    ]
    assert hier == ["Fovea|Tier A", "Nature|Birds|Anatidae|Mallard"]


def test_keywords_are_escaped() -> None:
    text = render(Sidecar(keywords=["A&B|C<D>"]))
    parse_description(text)
    assert "A&amp;B" in text


def test_empty_sidecar_renders_without_bags() -> None:
    desc = parse_description(render(Sidecar()))
    assert len(list(desc)) == 0


def test_invalid_values_rejected() -> None:
    with pytest.raises(ValueError):
        Sidecar(rating=6)
    with pytest.raises(ValueError):
        Sidecar(label="green")


def test_write_sidecar(tmp_path: Path) -> None:
    img = tmp_path / "IMG_0001.CR3"
    img.write_bytes(b"x")
    out = write_sidecar(img, Sidecar(rating=3))
    assert out == tmp_path / "IMG_0001.xmp"
    assert 'xmp:Rating="3"' in out.read_text()
