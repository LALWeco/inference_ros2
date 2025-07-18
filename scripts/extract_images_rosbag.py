import os
import argparse
from pathlib import Path
import cv2
from cv_bridge import CvBridge
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

cv_bridge = CvBridge()

def read_messages(input_bag: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        try:
            msg_type = get_message(typename(topic))
            msg = deserialize_message(data, msg_type)
            yield topic, msg, timestamp
        except (ImportError, ModuleNotFoundError):
            continue
    del reader

def extract_images_from_bag(bag_path, output_dir, topic, nth=1):
    bag_name = bag_path.stem
    output_bag_dir = output_dir / f"{bag_name}_images"
    output_bag_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    for i, (msg_topic, msg, timestamp) in enumerate(read_messages(str(bag_path))):
        if msg_topic == topic:
            if i % nth == 0:
                try:
                    cv_image = cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    image_filename = os.path.join(output_bag_dir, f'{bag_name}_{timestamp}.png')
                    cv2.imwrite(image_filename, cv_image)
                    image_count += 1
                except Exception as e:
                    print(f"Failed to convert or save image at {timestamp}: {e}")
    print(f"Extracted {image_count} images from {bag_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract every nth image from all ros2 mcap bag files in all subfolders of a directory.")
    parser.add_argument("input_folder", type=str, help="Root folder containing subfolders with mcap bag files")
    parser.add_argument("output_folder", type=str, help="Folder to save extracted images")
    parser.add_argument("--topic", type=str, required=True, help="Image topic to extract from")
    parser.add_argument("--nth", type=int, default=1, help="Extract every nth image (default: 1)")
    args = parser.parse_args()

    input_dir = Path(args.input_folder)
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through all subfolders and find .mcap files
    for subfolder in input_dir.iterdir():
        if subfolder.is_dir():
            for bag_file in subfolder.glob("*.mcap"):
                print(f"Processing {bag_file} ...")
                extract_images_from_bag(bag_file, output_dir, args.topic, args.nth)

if __name__ == "__main__":
    main()