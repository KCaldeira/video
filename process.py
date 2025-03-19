# video to midi code
"""
prompt for ChatGTP:

Can you write some python code that will process every Nth frame of a video file, and produce 
a midi file that will be a time series indicating the average grey-scale intensity of the frame. 
Assume that this will control the volume of a fader in a DAW.
"""

import cv2
import numpy as np
from mido import Message, MidiFile, MidiTrack

def video_to_midi(video_path, output_midi_path, nth_frame=10, cc_number=7, channel=0):
    """
    Processes every Nth frame of a video, calculates average grayscale intensity,
    and writes a MIDI file with CC messages to control volume in a DAW.

    :param video_path: Path to the video file.
    :param output_midi_path: Output MIDI file path.
    :param nth_frame: Process every Nth frame.
    :param cc_number: MIDI CC number (default 7 for volume).
    :param channel: MIDI channel (0-15).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    midi_file = MidiFile()
    track = MidiTrack()
    midi_file.tracks.append(track)

    frame_count = 0
    time_tick = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % nth_frame == 0:
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_intensity = np.mean(gray_frame)

            # Map intensity (0-255) to MIDI CC range (0-127)
            midi_value = int((avg_intensity / 255) * 127)

            # Create and append MIDI CC message
            track.append(Message('control_change', control=cc_number, value=midi_value, channel=channel, time=time_tick))

            # Increment MIDI time
            time_tick = 100  # Adjust based on frame rate for real-time mapping

        frame_count += 1

    cap.release()
    
    # Save MIDI file
    midi_file.save(output_midi_path)
    print(f"MIDI file saved: {output_midi_path}")

# Example usage
video_to_midi("input_video.mp4", "output_volume_control.mid", nth_frame=10)
