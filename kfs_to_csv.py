import struct
import pandas as pd

def read_kfs_to_dataframe(filepath):
    """
    Read a KFS binary file and return a DataFrame with x and y columns.

    Args:
        filepath: Path to the .kfs file

    Returns:
        DataFrame with columns ['x', 'y']
    """
    with open(filepath, 'rb') as f:
        # Read the number of points (4 bytes)
        nb_points_data = f.read(4)
        nb_points = struct.unpack('<i', nb_points_data)[0]

        # Read each point (2 floats = 8 bytes per point)
        points = []
        for _ in range(nb_points):
            point_data = f.read(8)
            x, y = struct.unpack('<ff', point_data)
            points.append((x, y))

        # Convert list of tuples to DataFrame
        df = pd.DataFrame(points, columns=['x', 'y'])
        return df

# File paths (adjust these to your actual file locations)
speed_path = "data/input/N30_T7a_speed.kfs"
evrot3d_path = None  # Set to None if you don't have this file, or provide the path

# Read speed file
speed_df = read_kfs_to_dataframe(speed_path)
print(f"Read {len(speed_df)} points from speed file")
speed_df.to_csv("speed.csv", index=False)
print(f"Saved speed.csv")

# Read evrot3d file only if path is provided
if evrot3d_path is not None:
    evrot3d_df = read_kfs_to_dataframe(evrot3d_path)
    print(f"Read {len(evrot3d_df)} points from evrot3d file")
    evrot3d_df.to_csv("EvRot3D.csv", index=False)
    print(f"Saved EvRot3D.csv")
else:
    print("No evrot3d file specified, skipping")
