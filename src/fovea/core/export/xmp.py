from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

LRC_LABELS = ("Red", "Yellow", "Green", "Blue", "Purple")
FOVEA_NS = "https://github.com/JamesZhang22/fovea/xmp/1.0/"

TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
    xmlns:fovea="{fovea_ns}"{attrs}>
{bags}  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


@dataclass
class Sidecar:
    rating: int | None = None
    label: str | None = None
    keywords: list[str] = field(default_factory=list)
    fovea: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rating is not None and not 0 <= self.rating <= 5:
            raise ValueError(f"rating must be 0-5, got {self.rating}")
        if self.label is not None and self.label not in LRC_LABELS:
            raise ValueError(f"label must be one of {LRC_LABELS} for LrC to render it")


def sidecar_path(image_path: Path) -> Path:
    """Sidecar path with the image extension replaced, the only form LrC pairs up."""
    return image_path.with_suffix(".xmp")


def render(s: Sidecar) -> str:
    """Serialize to XMP, keywords go to both flat dc:subject and lr:hierarchicalSubject."""
    attrs = []
    if s.rating is not None:
        attrs.append(f'xmp:Rating="{s.rating}"')
    if s.label is not None:
        attrs.append(f'xmp:Label="{s.label}"')
    for key, value in s.fovea.items():
        attrs.append(f"fovea:{key}={quoteattr(str(value))}")
    attr_block = "".join(f"\n    {a}" for a in attrs)

    bags = ""
    if s.keywords:
        flat = [k.split("|")[-1] for k in s.keywords]
        bags = _bag("dc:subject", flat) + _bag("lr:hierarchicalSubject", s.keywords)
    return TEMPLATE.format(fovea_ns=FOVEA_NS, attrs=attr_block, bags=bags)


def write_sidecar(image_path: Path, s: Sidecar) -> Path:
    """Write the sidecar next to the image, returns the path written."""
    out = sidecar_path(image_path)
    out.write_text(render(s), encoding="utf-8")
    return out


def _bag(tag: str, items: list[str]) -> str:
    lis = "\n".join(f"     <rdf:li>{escape(i)}</rdf:li>" for i in items)
    return f"   <{tag}>\n    <rdf:Bag>\n{lis}\n    </rdf:Bag>\n   </{tag}>\n"
