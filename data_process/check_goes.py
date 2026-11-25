from pathlib import Path
import numpy as np
import rasterio

GOES_DIR = Path(
    "/stor/soteria/hydro/private/rls134/data/dke_bnf/GOES/processed"
)

files = sorted(GOES_DIR.glob("*.tif"))

if not files:
    raise SystemExit(f"No GeoTIFF files found in {GOES_DIR}")

results = []

for path in files:
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype("float32")
        data = data.filled(np.nan)

    # Apply the same temperature validity range used in preprocessing
    valid = np.isfinite(data) & (data >= 200) & (data <= 350)
    fraction = np.count_nonzero(valid) / valid.size

    results.append((fraction, path.name))

fractions = np.array([r[0] for r in results])

print("\n===== VALID-PIXEL SUMMARY =====")
print(f"Total files: {len(results)}")
print(f"Mean valid fraction: {fractions.mean():.3f}")
print(f"Minimum valid fraction: {fractions.min():.3f}")
print(f"Files with >=90% valid pixels: {np.sum(fractions >= 0.90)}")
print(f"Files with <90% valid pixels: {np.sum(fractions < 0.90)}")
print(f"Files with 0% valid pixels: {np.sum(fractions == 0)}")

print("\n===== 20 FILES WITH LOWEST VALID FRACTIONS =====")
for fraction, name in sorted(results)[:20]:
    print(f"{fraction:.3f}  {name}")

print("\n===== FILES MEETING THE 90% REQUIREMENT =====")
for fraction, name in sorted(results, reverse=True):
    if fraction >= 0.90:
        print(f"{fraction:.3f}  {name}")
