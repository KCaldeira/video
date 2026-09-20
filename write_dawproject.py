#!/usr/bin/env python3
"""
Render metric curves as DAWproject volume automation on group (buss) tracks.

Reads the same `{prefix}_values.csv` that write_midi.py reads and writes a
single .dawproject carrying every metric column in that CSV as a volume
automation envelope.

The CSV is the contract: every column becomes a track. To get fewer tracks,
generate fewer columns -- narrow metrics_processing (color_channels, metrics,
rank_types, inversions, filter_seconds, stretch_*) rather than filtering
here.

This stage is tempo-free. Automation times are wall-clock seconds, computed as
`frame_count_list / frames_per_second`, so the curves stay locked to the video
and changing the project tempo in Cubase cannot move them. The Transport tempo
written into the file is cosmetic; nothing positional depends on it.

Nothing here needs beats or ticks. The frame rate is read from the config
that process_video wrote next to the values CSV, which is authoritative: it
records the rate actually used to produce those rows. A project tempo is
written only if one is configured, purely as a cosmetic Transport value.

Note that a track's time base (musical vs linear) is NOT part of the DAWproject
schema; it is a DAW-side property applied when the track is created. In Cubase
it comes from Preferences > Editing > Default Track Time Type, so set that to
Linear before importing, or switch the tracks afterwards.

Metric values are 0-1 and are written as linear gain directly: 0 is -infinity
dB and 1.0 is 0 dB (unity, fader top).

Usage:
    python write_dawproject.py json/N48_dawproject.json
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import xmlschema

from curve_to_dawproject import (
    DAWPROJECT_VERSION,
    IdFactory,
    build_channel,
    build_metadata_xml,
    real_parameter,
    serialize,
)

# Cubase (tested 15.0.30) will not attach automation to a submix (group)
# channel on
# import. This was established by testing eight structural variants, including
# one matching Cubase's own DAWproject export byte for byte in shape (group
# busses as bare <Channel role="submix"> directly in <Structure>): the busses
# appear, the automation does not. Cubase's exporter likewise writes no
# automation for a group channel.
#
# Nor is this specific to Volume: Pan, Send/Volume and Equalizer/OutputGain on
# a submix, and Volume on a VCA, were all tested and none attach. Cubase
# refuses every automatable parameter on a submix or VCA channel.
#
# What does work is a Track with contentType="audio" whose channel has
# role="submix". Cubase then creates *two* things per metric: an audio track
# carrying the automation, and a like-named group buss. Copy the automation
# from the audio track to the group by hand in Cubase.
#
# So this is a deliberate workaround for a DAW limitation, not the structure
# the format calls for. Revisit if a later Cubase imports group automation.
TRACK_SHAPE = "audio track + submix channel per metric"

# The group/buss role. The mixerRole enumeration is
# regular|master|effect|submix|vca -- there is no "group", and an invalid value
# is imported by Cubase as an ordinary audio track rather than rejected, so the
# generated XML is schema-validated on every write.
GROUP_ROLE = "submix"

PROJECT_XSD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "schema", "dawproject", "Project.xsd")

APPLICATION_NAME = "write_dawproject"
APPLICATION_VERSION = "1.0"

# Metric values are 0-1 and map straight to linear gain, so the fader ceiling
# is unity: metric 1.0 == 0 dB.
VOLUME_MAX = 1.0

FRAME_COLUMN = "frame_count_list"

# Each column costs two Cubase tracks (the audio/buss pair), so a full
# unfiltered CSV would be thousands. Refuse rather than emit something
# unopenable; the fix is always to narrow metrics_processing.
DEFAULT_MAX_COLUMNS = 64


def build_project_xml(times, columns, values, tempo_bpm, interpolation):
    """
    Build the project.xml tree: one group buss per column, plus a master.

    :param times: sequence of automation times in seconds
    :param columns: list of column names, one group track each
    :param values: DataFrame holding the metric columns (0-1 linear gain)
    """
    ids = IdFactory()

    root = ET.Element("Project")
    root.set("version", DAWPROJECT_VERSION)

    application = ET.SubElement(root, "Application")
    application.set("name", APPLICATION_NAME)
    application.set("version", APPLICATION_VERSION)

    # Both Tempo and TimeSignature are optional. A tempo is written only when
    # one is configured: nothing positional depends on it, and stating a wrong
    # number is worse than stating none -- N48, for instance, runs a variable
    # tempo map, so no single value would be correct.
    transport = ET.SubElement(root, "Transport")
    if tempo_bpm is not None:
        real_parameter(transport, "Tempo", tempo_bpm, "bpm", 20.0, 666.0, "Tempo", ids)
    signature = ET.SubElement(transport, "TimeSignature")
    signature.set("denominator", "4")
    signature.set("numerator", "4")
    signature.set("id", ids.next())

    structure = ET.SubElement(root, "Structure")

    # Group tracks are written first and the master second, matching the order
    # DAWs use. Each group channel's destination is a forward reference to the
    # master channel, which XML IDREF resolution allows.
    # Each Track carries contentType="audio" *and* a submix channel, so Cubase
    # creates a pair per metric: an audio track holding the automation, and a
    # like-named group buss. See TRACK_SHAPE above for why.
    group_tracks = []
    for column in columns:
        track = ET.SubElement(structure, "Track")
        track.set("contentType", "audio")
        track.set("loaded", "true")
        track.set("id", ids.next())
        track.set("name", column)
        group_tracks.append(track)

    master_track = ET.SubElement(structure, "Track")
    master_track.set("contentType", "audio")
    master_track.set("loaded", "true")
    master_track.set("id", ids.next())
    master_track.set("name", "Master")
    master_channel, _ = build_channel(master_track, ids, "master", VOLUME_MAX)
    master_channel_id = master_channel.get("id")

    volume_ids = []
    for track in group_tracks:
        _, volume = build_channel(
            track, ids, GROUP_ROLE, VOLUME_MAX, destination=master_channel_id
        )
        volume_ids.append(volume.get("id"))

    arrangement = ET.SubElement(root, "Arrangement")
    arrangement.set("id", ids.next())
    outer_lanes = ET.SubElement(arrangement, "Lanes")
    outer_lanes.set("timeUnit", "seconds")
    outer_lanes.set("id", ids.next())

    # The busses are empty by design: no Clips, only automation. Sources are
    # routed into them by hand in the DAW.
    for track, column, volume_id in zip(group_tracks, columns, volume_ids):
        track_id = track.get("id")
        track_lanes = ET.SubElement(outer_lanes, "Lanes")
        track_lanes.set("track", track_id)
        track_lanes.set("id", ids.next())

        points = ET.SubElement(track_lanes, "Points")
        points.set("timeUnit", "seconds")
        points.set("unit", "linear")
        points.set("track", track_id)
        points.set("id", ids.next())
        target = ET.SubElement(points, "Target")
        target.set("parameter", volume_id)

        for time, gain in zip(times, values[column]):
            point = ET.SubElement(points, "RealPoint")
            point.set("time", f"{time:.6f}")
            point.set("value", f"{gain:.9f}")
            point.set("interpolation", interpolation)

    ET.SubElement(root, "Scenes")
    return root


def validate_project_xml(project_xml):
    """
    Validate against the vendored DAWproject schema.

    Not optional: DAWs tend to accept invalid enumeration values and silently
    do the wrong thing with them, so a schema error here is the only place the
    mistake is visible.
    """
    errors = list(xmlschema.XMLSchema(PROJECT_XSD).iter_errors(
        project_xml.decode("utf-8")))
    if errors:
        detail = "\n".join(f"  {error.reason}" for error in errors[:5])
        raise ValueError(
            f"generated project.xml is not schema-valid "
            f"({len(errors)} error(s)):\n{detail}"
        )


def read_frames_per_second(stage1_config):
    """
    Read the frame rate from the config process_video wrote beside the CSV.

    That file is authoritative: it records the rate actually used to produce
    the rows, so it cannot drift from them the way a pipeline config can.
    """
    if not os.path.exists(stage1_config):
        raise FileNotFoundError(
            f"No video-stage config at {stage1_config}; it records the frame "
            f"rate the values CSV was produced with"
        )
    with open(stage1_config) as handle:
        recorded = json.load(handle)
    if "frames_per_second" not in recorded:
        raise KeyError(f"{stage1_config} has no 'frames_per_second'")
    return recorded["frames_per_second"]


def render_metrics_dawproject(values_csv, output_path, max_columns,
                              frames_per_second, tempo_bpm, interpolation, title):
    """Read the values CSV and write the .dawproject archive."""
    csv = pd.read_csv(values_csv)

    columns = [c for c in csv.columns if c != FRAME_COLUMN]
    if not columns:
        raise ValueError(f"{values_csv} holds no metric columns")
    if len(columns) > max_columns:
        raise ValueError(
            f"{values_csv} holds {len(columns)} metric columns, which would "
            f"make {2 * len(columns)} Cubase tracks (each column is an audio "
            f"track plus a buss). The limit is {max_columns}; narrow "
            f"metrics_processing (color_channels, metrics, rank_types, "
            f"inversions, filter_seconds, stretch_*) or raise "
            f"dawproject.max_columns."
        )


    # Automation times are wall-clock seconds derived from the video frame
    # numbers, which is what keeps the curves locked to the video.
    times = csv[FRAME_COLUMN] / frames_per_second

    project_xml = serialize(
        build_project_xml(times, columns, csv, tempo_bpm, interpolation)
    )
    validate_project_xml(project_xml)
    metadata_xml = serialize(
        build_metadata_xml(
            title,
            f"Volume automation for {len(columns)} metrics from "
            f"{os.path.basename(values_csv)}",
        )
    )

    with zipfile.ZipFile(output_path, "w") as archive:
        archive.writestr("project.xml", project_xml, zipfile.ZIP_DEFLATED)
        archive.writestr("metadata.xml", metadata_xml, zipfile.ZIP_DEFLATED)

    size = os.path.getsize(output_path)
    print(f"Wrote {output_path} ({size / 1e6:.2f} MB)")
    print(f"  {len(columns)} audio tracks (automation) + {len(columns)} group "
          f"busses, {len(times)} points each")
    print("  Cubase will not import automation onto a group: copy each lane")
    print("  from the audio track to its like-named buss by hand.")
    print(f"  Time range: 0.000 to {times.iloc[-1]:.3f} seconds")
    print("Import in Cubase with File > Import > DAWproject.")


def write_dawproject_from_config(config):
    """Entry point used by run_video_processing.py and by __main__."""
    video = config.get("video", {})
    dawproject = config.get("dawproject", {})

    video_name = video.get("video_name")
    preset = config.get("video_processing", {}).get("optical_flow", {}).get(
        "preset", "default")

    # ticks_per_beat is not read at all, and beats_per_minute only supplies a
    # cosmetic Transport tempo. A config may omit the timing section entirely.
    beats_per_minute = config.get("timing", {}).get("beats_per_minute")
    tempo_bpm = (float(beats_per_minute)
                 if isinstance(beats_per_minute, (int, float)) else None)

    if "columns" in dawproject:
        raise ValueError(
            "dawproject.columns is no longer used: every column in the values "
            "CSV is exported. Narrow metrics_processing instead.")
    max_columns = dawproject.get("max_columns", DEFAULT_MAX_COLUMNS)
    interpolation = dawproject.get("interpolation", "linear")

    # video_name may contain a subdirectory (e.g. "N44/N44_testgi2"): the full
    # path names the output directory, but file names use only the basename.
    output_dir = f"data/output/{video_name}_{preset}"
    name_prefix = f"{os.path.basename(video_name)}_{preset}"

    frames_per_second = read_frames_per_second(
        f"{output_dir}/{name_prefix}_config.json")

    values_csv = f"{output_dir}/{name_prefix}_values.csv"
    if not os.path.exists(values_csv):
        raise FileNotFoundError(f"No metrics values CSV at {values_csv}")
    print(f"Using metrics values CSV: {values_csv}")

    render_metrics_dawproject(
        values_csv,
        f"{output_dir}/{name_prefix}.dawproject",
        max_columns,
        frames_per_second,
        tempo_bpm,
        interpolation,
        name_prefix,
    )


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    with open(sys.argv[1]) as handle:
        config = json.load(handle)
    write_dawproject_from_config(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
