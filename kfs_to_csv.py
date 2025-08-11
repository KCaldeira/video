import struct
import pandas as pd

def read_kfs_to_dataframe(filepath):
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
speed_path = "speed.kfs"
evrot3d_path = "EvRot3D.kfs"

# Read binary files into DataFrames
speed_df = read_kfs_to_dataframe(speed_path)
evrot3d_df = read_kfs_to_dataframe(evrot3d_path)

# Save as CSV
speed_df.to_csv("speed.csv", index=False)
evrot3d_df.to_csv("EvRot3D.csv", index=False)
