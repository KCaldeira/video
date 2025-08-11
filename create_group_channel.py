import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    return minidom.parseString(rough_string).toprettyxml(indent="  ")

def generate_cubase_group_file(Start, Data2, Name, base_id=3000000000):
    """
    Generate a Cubase group track XML with MMidiController automation.
    HrData is normalized for Cubase 14.
    """
    tracklist = ET.Element("tracklist2")
    track_list = ET.SubElement(tracklist, "list", {"name": "track", "type": "obj"})

    device_event = ET.SubElement(track_list, "obj", {
        "class": "MDeviceTrackEvent", "ID": str(base_id)
    })
    base_id += 1
    ET.SubElement(device_event, "int", {"name": "Flags", "value": "1"})
    ET.SubElement(device_event, "float", {"name": "Start", "value": "0"})
    ET.SubElement(device_event, "float", {"name": "Length", "value": "3840"})

    node = ET.SubElement(device_event, "obj", {
        "class": "MListNode", "name": "Node", "ID": str(base_id)
    })
    base_id += 1
    ET.SubElement(node, "string", {"name": "Name", "value": Name, "wide": "true"})

    domain = ET.SubElement(node, "member", {"name": "Domain"})
    ET.SubElement(domain, "int", {"name": "Type", "value": "0"})

    tempo_track = ET.SubElement(domain, "obj", {
        "class": "MTempoTrackEvent", "name": "Tempo Track", "ID": str(base_id)
    })
    base_id += 1
    tempo_event_list = ET.SubElement(tempo_track, "list", {"name": "TempoEvent", "type": "obj"})

    tempo_event = ET.SubElement(tempo_event_list, "obj", {"class": "MTempoEvent", "ID": str(base_id)})
    base_id += 1
    ET.SubElement(tempo_event, "float", {"name": "BPM", "value": "120"})
    ET.SubElement(tempo_event, "float", {"name": "PPQ", "value": "0"})
    ET.SubElement(tempo_event, "float", {"name": "RehearsalTempo", "value": "64"})
    ET.SubElement(tempo_event, "int", {"name": "RehearsalMode", "value": "1"})

    additional = ET.SubElement(tempo_event, "member", {"name": "Additional Attributes"})
    ET.SubElement(additional, "int", {"name": "TTlB", "value": "40"})
    ET.SubElement(additional, "int", {"name": "TTuB", "value": "270"})
    ET.SubElement(additional, "int", {"name": "TLID", "value": "1"})

    events_list = ET.SubElement(node, "list", {"name": "Events", "type": "obj"})

    midi_event = ET.SubElement(events_list, "obj", {"class": "MMidiPartEvent", "ID": str(base_id)})
    base_id += 1
    ET.SubElement(midi_event, "float", {"name": "Start", "value": "0"})
    ET.SubElement(midi_event, "float", {"name": "Length", "value": "3840"})

    midi_part = ET.SubElement(midi_event, "obj", {
        "class": "MMidiPart", "name": "Node", "ID": str(base_id)
    })
    base_id += 1
    ET.SubElement(midi_part, "string", {"name": "Name", "value": Name})

    domain2 = ET.SubElement(midi_part, "member", {"name": "Domain"})
    ET.SubElement(domain2, "int", {"name": "Type", "value": "0"})
    ET.SubElement(domain2, "obj", {"name": "Tempo Track", "ID": "1153034128"})
    ET.SubElement(domain2, "obj", {"name": "Signature Track", "ID": "1117328832"})

    controller_list = ET.SubElement(midi_part, "list", {"name": "Events", "type": "obj"})

    for s, d2 in zip(Start, Data2):
        hr = d2 / 127.0  # Normalize for Cubase 14
        ctrl = ET.SubElement(controller_list, "obj", {"class": "MMidiController", "ID": str(base_id)})
        base_id += 1
        ET.SubElement(ctrl, "float", {"name": "Start", "value": str(s)})
        ET.SubElement(ctrl, "int", {"name": "Channel", "value": "7"})
        ET.SubElement(ctrl, "int", {"name": "Data1", "value": "1"})
        ET.SubElement(ctrl, "int", {"name": "Data2", "value": str(d2)})
        ET.SubElement(ctrl, "float", {"name": "HrData", "value": str(hr)})
        ET.SubElement(ctrl, "int", {"name": "PreNoteOn", "value": "0"})
        ET.SubElement(ctrl, "int", {"name": "OneShotDef", "value": "0"})

    return prettify(tracklist)

# Entry point
if __name__ == "__main__":
    # Default test data
    Start = [0, 480, 960, 1440]
    Data2 = [0, 64, 100, 127]
    default_name = "Generated Group"
    default_output = "GeneratedGroup.trackarchives"

    # Optional CLI args
    name = sys.argv[1] if len(sys.argv) > 1 else default_name
    out_file = sys.argv[2] if len(sys.argv) > 2 else default_output

    print(f"Generating Cubase 14 group track '{name}' → {out_file}")
    xml = generate_cubase_group_file(Start, Data2, name)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(xml)

    print("Done.")
