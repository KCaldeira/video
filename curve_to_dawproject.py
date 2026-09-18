#!/usr/bin/env python3
"""
Package a gain curve and its audio into a DAWproject file for Cubase.

Stage 2 of the audio auto-leveling workflow. Takes the decimated automation
points written by audio_level_curve.py and builds a .dawproject: a ZIP holding
project.xml, metadata.xml, and the audio itself, with the curve attached to the
track's volume parameter as an automation envelope.

The source audio is copied into the archive byte for byte -- it is never
decoded, re-encoded, resampled, or requantized -- so Cubase receives the
original file plus a fader move, which is lossless by construction.

Everything is expressed in seconds. The DAWproject schema makes Transport and
Tempo optional and gives every timeline its own timeUnit, so no position in the
generated file depends on tempo.

Usage:
    python curve_to_dawproject.py POINTS_CSV AUDIO_FILE [options]

Examples:
    python curve_to_dawproject.py \\
        "data/output/N47 2026-09-17-04_c28_gau15_pk0.1_gain_points.csv" \\
        "data/input/N47 2026-09-17-04.wav"
"""

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from xml.dom import minidom

import numpy as np
import soundfile as sf

DESCRIPTION = """\
Package a gain curve and its audio into a DAWproject file for Cubase.

Reads the decimated automation points from audio_level_curve.py and writes a
.dawproject containing one audio track: the source audio embedded unchanged,
placed at time zero, with the gain curve attached to the track's volume
parameter as a linear automation envelope.

Cubase imports this with File > Import > DAWproject.
"""

EPILOG = """\
notes:
  The audio is stored in the archive byte for byte, so no quality is lost. The
  archive is therefore about as large as the source file.

  Automation values are linear gain, matching the unit declared on the track's
  Volume parameter. --volume-max-db sets that parameter's ceiling; it must be
  at least as large as the largest gain in the curve, and defaults to Cubase's
  own +12 dB fader maximum.

  Times are in seconds throughout. The file carries a Transport/Tempo element
  because Cubase always has a project tempo, but nothing positional depends on
  it.

examples:
  python curve_to_dawproject.py curve_gain_points.csv song.wav
  python curve_to_dawproject.py curve_gain_points.csv song.wav --external-audio
  python curve_to_dawproject.py curve_gain_points.csv song.wav --track-name "Lead Vox"
"""

DAWPROJECT_VERSION = "1.0"
APPLICATION_NAME = "audio_level_curve"
APPLICATION_VERSION = "1.0"

# Cubase's fader maximum, and the default ceiling for the Volume parameter.
DEFAULT_VOLUME_MAX_DB = 12.0

DEFAULT_TEMPO_BPM = 120.0


class IdFactory:
    """Sequential XML ids, matching the idN convention the format uses."""

    def __init__(self):
        self.count = 0

    def next(self):
        value = f"id{self.count}"
        self.count += 1
        return value


def real_parameter(parent, tag, value, unit, minimum, maximum, name, ids):
    element = ET.SubElement(parent, tag)
    element.set("max", f"{maximum:.6f}")
    element.set("min", f"{minimum:.6f}")
    element.set("unit", unit)
    element.set("value", f"{value:.6f}")
    element.set("id", ids.next())
    element.set("name", name)
    return element


def build_channel(parent, ids, role, volume_max, destination=None, audio_channels=2):
    channel = ET.SubElement(parent, "Channel")
    channel.set("audioChannels", str(audio_channels))
    if destination is not None:
        channel.set("destination", destination)
    channel.set("role", role)
    channel.set("solo", "false")
    channel.set("id", ids.next())

    mute = ET.SubElement(channel, "Mute")
    mute.set("value", "false")
    mute.set("id", ids.next())
    mute.set("name", "Mute")

    real_parameter(channel, "Pan", 0.5, "normalized", 0.0, 1.0, "Pan", ids)
    volume = real_parameter(
        channel, "Volume", 1.0, "linear", 0.0, volume_max, "Volume", ids
    )
    return channel, volume


def build_project_xml(
    point_times, point_gain_linear, audio_path_in_zip, info, track_name,
    volume_max_db, interpolation, tempo_bpm, external_path,
):
    ids = IdFactory()
    volume_max = 10.0 ** (volume_max_db / 20.0)

    root = ET.Element("Project")
    root.set("version", DAWPROJECT_VERSION)

    application = ET.SubElement(root, "Application")
    application.set("name", APPLICATION_NAME)
    application.set("version", APPLICATION_VERSION)

    transport = ET.SubElement(root, "Transport")
    real_parameter(transport, "Tempo", tempo_bpm, "bpm", 20.0, 666.0, "Tempo", ids)
    signature = ET.SubElement(transport, "TimeSignature")
    signature.set("denominator", "4")
    signature.set("numerator", "4")
    signature.set("id", ids.next())

    structure = ET.SubElement(root, "Structure")

    # The audio track is written first and the master second, matching the
    # order DAWs use. The audio channel's destination is a forward reference to
    # the master channel, which XML IDREF resolution allows.
    audio_track = ET.SubElement(structure, "Track")
    audio_track.set("contentType", "audio")
    audio_track.set("loaded", "true")
    audio_track_id = ids.next()
    audio_track.set("id", audio_track_id)
    audio_track.set("name", track_name)

    master_track = ET.SubElement(structure, "Track")
    master_track.set("contentType", "audio")
    master_track.set("loaded", "true")
    master_track.set("id", ids.next())
    master_track.set("name", "Master")
    master_channel, _ = build_channel(master_track, ids, "master", volume_max)
    master_channel_id = master_channel.get("id")

    _, volume = build_channel(
        audio_track, ids, "regular", volume_max, destination=master_channel_id,
        audio_channels=info.channels,
    )
    volume_id = volume.get("id")

    duration = info.frames / info.samplerate

    arrangement = ET.SubElement(root, "Arrangement")
    arrangement.set("id", ids.next())
    outer_lanes = ET.SubElement(arrangement, "Lanes")
    outer_lanes.set("timeUnit", "seconds")
    outer_lanes.set("id", ids.next())

    track_lanes = ET.SubElement(outer_lanes, "Lanes")
    track_lanes.set("track", audio_track_id)
    track_lanes.set("id", ids.next())

    clips = ET.SubElement(track_lanes, "Clips")
    clips.set("id", ids.next())
    clip = ET.SubElement(clips, "Clip")
    clip.set("time", "0.0")
    clip.set("duration", f"{duration:.9f}")
    clip.set("contentTimeUnit", "seconds")
    clip.set("playStart", "0.0")
    clip.set("name", track_name)

    audio = ET.SubElement(clip, "Audio")
    audio.set("channels", str(info.channels))
    audio.set("duration", f"{duration:.9f}")
    audio.set("sampleRate", str(info.samplerate))
    audio.set("id", ids.next())
    file_element = ET.SubElement(audio, "File")
    if external_path is None:
        file_element.set("path", audio_path_in_zip)
    else:
        file_element.set("path", external_path)
        file_element.set("external", "true")

    points = ET.SubElement(track_lanes, "Points")
    points.set("timeUnit", "seconds")
    points.set("unit", "linear")
    points.set("track", audio_track_id)
    points.set("id", ids.next())
    target = ET.SubElement(points, "Target")
    target.set("parameter", volume_id)
    for time, gain in zip(point_times, point_gain_linear):
        point = ET.SubElement(points, "RealPoint")
        point.set("time", f"{time:.6f}")
        point.set("value", f"{gain:.9f}")
        point.set("interpolation", interpolation)

    ET.SubElement(root, "Scenes")
    return root


def build_metadata_xml(title, comment):
    root = ET.Element("MetaData")
    ET.SubElement(root, "Title").text = title
    ET.SubElement(root, "Comment").text = comment
    return root


def serialize(element):
    raw = ET.tostring(element, encoding="utf-8")
    pretty = minidom.parseString(raw).toprettyxml(indent="  ", encoding="UTF-8")
    return pretty


def main():
    parser = argparse.ArgumentParser(
        prog="curve_to_dawproject.py",
        description=DESCRIPTION,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "points_csv",
        help="automation points from audio_level_curve.py "
        "(columns time_s, gain_db, gain_linear)",
    )
    parser.add_argument(
        "audio_file",
        help="the audio the curve was computed from; embedded unchanged",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output .dawproject path (default: the points file's name with "
        "the _gain_points suffix replaced, in data/output/)",
    )
    parser.add_argument(
        "--track-name",
        default=None,
        help="name of the track in the project (default: the audio file's name)",
    )
    parser.add_argument(
        "--volume-max-db",
        type=float,
        default=DEFAULT_VOLUME_MAX_DB,
        help="ceiling declared on the track's Volume parameter; must cover the "
        "largest gain in the curve (default: %(default)s, Cubase's fader maximum)",
    )
    parser.add_argument(
        "--interpolation",
        choices=("linear", "hold"),
        default="linear",
        help="how the DAW should interpolate between points (default: %(default)s)",
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=DEFAULT_TEMPO_BPM,
        help="project tempo written to Transport; nothing positional depends "
        "on it, since all times are in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--external-audio",
        action="store_true",
        help="reference the audio by absolute path instead of embedding it, "
        "producing a tiny file; the path must resolve on the importing machine",
    )
    parser.add_argument(
        "--validate",
        metavar="PROJECT_XSD",
        default=None,
        help="validate the generated project.xml against this copy of the "
        "DAWproject schema before writing",
    )

    args = parser.parse_args()

    if not os.path.exists(args.points_csv):
        parser.error(f"points file not found: {args.points_csv}")
    if not os.path.exists(args.audio_file):
        parser.error(f"audio file not found: {args.audio_file}")

    table = np.genfromtxt(args.points_csv, delimiter=",", names=True)
    point_times = np.atleast_1d(table["time_s"])
    point_gain_linear = np.atleast_1d(table["gain_linear"])
    if len(point_times) < 2:
        raise ValueError(f"need at least two automation points, got {len(point_times)}")
    if np.any(np.diff(point_times) < 0):
        raise ValueError("automation point times must be non-decreasing")

    info = sf.info(args.audio_file)
    volume_max = 10.0 ** (args.volume_max_db / 20.0)
    largest = float(point_gain_linear.max())
    if largest > volume_max:
        raise ValueError(
            f"the curve reaches {20.0 * np.log10(largest):+.2f} dB but the Volume "
            f"parameter is capped at {args.volume_max_db:+.2f} dB; raise "
            f"--volume-max-db to at least {20.0 * np.log10(largest):.2f}"
        )

    audio_basename = os.path.basename(args.audio_file)
    track_name = args.track_name or os.path.splitext(audio_basename)[0]
    audio_path_in_zip = f"audio/{audio_basename}"
    external_path = os.path.abspath(args.audio_file) if args.external_audio else None

    if args.output is None:
        stem = os.path.basename(args.points_csv)
        for suffix in ("_gain_points.csv", ".csv"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        output_path = os.path.join("data", "output", f"{stem}.dawproject")
    else:
        output_path = args.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print("=" * 60)
    print("CURVE TO DAWPROJECT")
    print("=" * 60)
    print(f"  Points:           {args.points_csv} ({len(point_times)} points)")
    print(f"  Audio:            {args.audio_file}")
    print(f"  Format:           {info.samplerate} Hz, {info.channels} ch, "
          f"{info.subtype}, {info.frames / info.samplerate:.1f} s")
    print(f"  Gain range:       {20.0 * np.log10(point_gain_linear.min()):+.2f} to "
          f"{20.0 * np.log10(largest):+.2f} dB")
    print(f"  Volume ceiling:   {args.volume_max_db:+.2f} dB (linear {volume_max:.6f})")
    print(f"  Audio placement:  {'external reference' if args.external_audio else 'embedded'}")
    print(f"  Output:           {output_path}")
    print()

    project = build_project_xml(
        point_times, point_gain_linear, audio_path_in_zip, info, track_name,
        args.volume_max_db, args.interpolation, args.tempo, external_path,
    )
    project_xml = serialize(project)
    metadata_xml = serialize(
        build_metadata_xml(
            track_name,
            f"Volume automation from {os.path.basename(args.points_csv)}",
        )
    )

    if args.validate is not None:
        import xmlschema

        print(f"Validating against {args.validate}")
        schema = xmlschema.XMLSchema(args.validate)
        schema.validate(project_xml.decode("utf-8"))
        print("  project.xml is schema-valid\n")

    with zipfile.ZipFile(output_path, "w") as archive:
        archive.writestr("project.xml", project_xml, zipfile.ZIP_DEFLATED)
        archive.writestr("metadata.xml", metadata_xml, zipfile.ZIP_DEFLATED)
        if not args.external_audio:
            # Stored, not deflated: audio does not compress and streaming the
            # copy keeps a large file out of memory.
            with open(args.audio_file, "rb") as source, archive.open(
                zipfile.ZipInfo(audio_path_in_zip), "w"
            ) as target:
                shutil.copyfileobj(source, target, length=1 << 20)

    size = os.path.getsize(output_path)
    print(f"Wrote {output_path} ({size / 1e6:.1f} MB)")
    print("Import in Cubase with File > Import > DAWproject.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
