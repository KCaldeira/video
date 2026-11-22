#!/usr/bin/env bash
# Level one audio file and put the results beside it.
#
#   ./level_dropped.sh FILE [audio_level_curve.py flags...]
#
# FILE may be a Windows path (F:\Cubase Projects\N59\Mixdown\mix.wav, or a
# \\wsl.localhost\... path) or a WSL path. This is the WSL side of
# windows/Level audio.bat, which runs it for each file dropped on its icon.
#
# Runs audio_level_curve.py, then curve_to_dawproject.py, with outputs in
# data/output/ as usual, then copies the three files you need in Cubase into
# the source file's folder:
#   <stem>_<tag>_leveled.wav   the finished result (ride plus makeup gain)
#   <stem>_<tag>.dawproject    the source audio with the ride as automation
#   <stem>_<tag>_plot.png      loudness before and after, and the gain curve
# The tag encodes the settings, so the source file is never overwritten.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 FILE [audio_level_curve.py flags...]" >&2
    exit 2
fi

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo"

src="$1"
shift
case "$src" in
    [A-Za-z]:\\*|\\\\*) src="$(wslpath -u "$src")" ;;
esac
if [ ! -f "$src" ]; then
    echo "ERROR: no such file: $src" >&2
    exit 1
fi
dest_dir="$(dirname "$src")"

python=".venv/bin/python"
out_dir="data/output"
mkdir -p "$out_dir"

marker="$(mktemp)"
trap 'rm -f "$marker"' EXIT

nice -n 19 "$python" audio_level_curve.py "$src" "$@"

# The run's points file is the only one written since the marker.
mapfile -t points < <(find "$out_dir" -maxdepth 1 -name '*_gain_points.csv' -newer "$marker")
if [ "${#points[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one new *_gain_points.csv in $out_dir, found ${#points[@]}:" >&2
    printf '  %s\n' "${points[@]}" >&2
    exit 1
fi
prefix="${points[0]%_gain_points.csv}"

echo
echo "Packaging DAWproject"
nice -n 19 "$python" curve_to_dawproject.py "${points[0]}" "$src" --output "$prefix.dawproject"

echo
echo "Copying to $dest_dir"
for f in "${prefix}_leveled.wav" "$prefix.dawproject" "${prefix}_plot.png"; do
    cp "$f" "$dest_dir/"
    echo "  $(basename "$f")"
done
echo
echo "Done. In Cubase: File > Import > Audio File for the _leveled.wav, which is"
echo "finished (ride plus makeup gain). Or File > Import > DAWproject for the ride"
echo "as editable automation; that one still needs the clip gain reported above."
