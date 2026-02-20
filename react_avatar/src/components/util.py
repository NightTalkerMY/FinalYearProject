# from pydub import AudioSegment
# import os

# # Get base directory of this script
# base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../public/audios'))

# # Input/output filenames
# input_file = "444.mp3"
# output_file = "444.ogg"

# # Full paths
# input_path = os.path.join(base_dir, input_file)
# output_path = os.path.join(base_dir, output_file)

# # Convert
# audio = AudioSegment.from_mp3(input_path)
# audio.export(output_path, format="ogg")

# # Delete original MP3 after successful conversion
# if os.path.exists(output_path):  # Make sure .ogg was created
#     os.remove(input_path)
#     print(f"Deleted original file: {input_file}")

# print(f"Converted '{input_file}' to '{output_file}' at:", output_path)

import subprocess
import os

# Path to rhubarb.exe relative to this script
rhubarb_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/components/rhubarb/rhubarb.exe'))

# Define input/output paths
input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../public/audios/1.ogg'))
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../public/json/1.json'))

# Command
command = [
    rhubarb_path,
    "-f", "json",
    input_path,
    "-o", output_path
]

try:
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

except subprocess.CalledProcessError as e:
    print("Error occurred:", e)
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
