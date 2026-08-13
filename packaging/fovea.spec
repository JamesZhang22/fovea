# PyInstaller spec: fovea.app -> dist/fovea.app
# Build: uv run pyinstaller packaging/fovea.spec --noconfirm

from pathlib import Path

ROOT = Path(SPECPATH).parent

a = Analysis(
    [str(ROOT / "src" / "fovea" / "app" / "main.py")],
    pathex=[str(ROOT / "src")],
    datas=[
        (str(ROOT / "ui" / "dist"), "ui/dist"),
        (str(ROOT / "models" / "eye.onnx"), "models"),
        (str(ROOT / "models" / "eye.onnx.data"), "models"),
        (str(ROOT / "models" / "bird.onnx"), "models"),
        (str(ROOT / "models" / "focus.onnx"), "models"),
        (str(ROOT / "models" / "focus.onnx.data"), "models"),
    ],
    hiddenimports=[
        "uvicorn.logging",
        "uvicorn.loops.auto",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan.on",
    ],
    excludes=["torch", "torchvision", "rfdetr", "transformers", "timm", "onnx", "matplotlib"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="fovea",
    console=False,
)

coll = COLLECT(exe, a.binaries, a.datas, name="fovea")

app = BUNDLE(
    coll,
    name="fovea.app",
    bundle_identifier="dev.jameszhang.fovea",
    info_plist={
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
    },
)
