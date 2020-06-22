# 6/16/2020
# plee@mbari.org
# Copyright Peyton Lee, MBARI 2020

"""Reads and converts JSON computer vision output to a CVAT-compatible XML format."""

import json
import xml.etree.ElementTree as ET
import xml.dom.minidom
import glob
import sys
from datetime import datetime, timezone


def read_JSON_annotations(files: [str]) -> {}:
    """ Reads the set of JSON annotations into a map from the uuid's to the Visual
    Events.

    Parameters
    ----------
    files : an array of JSON files to read from. These should be from the same
             annotated video. Each JSON file should be formatted to the MBARI
             standard (bounding box, class_index, class_name, uuid, etc.)

    Returns
    -------
    dict
        A dictionary, where each uuid is a key that maps to a list of the
        VisualEvents (parsed as a dictionary) for that uuid.
        {uuid: [uuid_frame1, uuid_frame2, ...], ...}
    """
    ret = {}
    for file in files:
        # Read in the file JSON
        with open(file) as f:
            filedata = json.load(f)

        # Loop through the array containing the visual events
        for i in range(0, len(filedata[1])):
            # Extract the event data, which contains the bounding box, class, etc.
            eventdata = filedata[1][i][1]
            uuid = eventdata["uuid"]
            # If the uuid is not in the dictionary, add it.
            if uuid not in ret:
                ret[uuid] = []
            # Append the eventdata to the array for that uuid.
            ret[uuid].append(eventdata)
    return ret

# Class for static variables/fields.


class MBARI_Data:
    # Maps the attribute names to their types, written as strings.
    attribute_types = {'class_index': 'number',
                       'class_name': 'text',
                       'confidence': 'number',
                       'occluded_pixels': 'number',
                       'surprise': 'number',
                       'uuid': 'text'}
    # Whether this attribute can be changed over the sequence.
    attribute_mutability = {'class_index': 'False',
                            'class_name': 'False',
                            'confidence': 'True',
                            'occluded_pixels': 'True',
                            'surprise': 'True',
                            'uuid': 'False'}


def build_sub_element(parent: ET.Element, name: str, text="",  attrib={}) -> ET.Element:
    """ Builds an XML SubElement with the given parent, name, and text.
    """
    ret = ET.SubElement(parent, name, attrib=attrib)
    ret.text = text
    return ret


def build_box_from_framedata(parent_element: ET.Element,
                             framedata: {}, frame_override=-1, outside=0,
                             occluded=0, keyframe=1):
    """ Creates and returns a box XML element with coordinates and attributes
        given by the framedata.

        Parameters
        ----------
        parent_element: Element
          The parent element for the XML box.
        framedata : {}
          The dictionary data for the given frame, as imported from MBARI standard
          data.
        frame_override: int
          (optional) Used as the frame number of the framedata.
        outside : int
          (optional) The value of the outside flag. 0 by default.
        occluded : int
          (optional) The value of the occluded flag, 0 by default.
        keyframe : int
          (optional) The value of the keyframe flag, 1 by default.

        Returns
        -------
        Element
          An XML Element of type box, with attribute subelements given by
          framedata and attributes given by framedata.
    """
    box = ET.SubElement(parent_element, "box")

    # Set the frame number with the override.
    if (frame_override == -1):
        box.attrib["frame"] = str(framedata["frame_num"] - 1)
    else:
        box.attrib["frame"] = str(frame_override)
    # Set the three flag values.
    box.attrib["outside"] = str(outside)
    box.attrib["occluded"] = str(occluded)
    box.attrib["keyframe"] = str(keyframe)

    # Top left x, y and bottom right x, y of the bounding box.
    box.attrib["xtl"] = str(framedata["bounding_box"]["x"]*2)
    box.attrib["ytl"] = str(framedata["bounding_box"]["y"]*2)
    box.attrib["xbr"] = str(framedata["bounding_box"]
                            ["x"]*2 + framedata["bounding_box"]["width"]*2)
    box.attrib["ybr"] = str(framedata["bounding_box"]
                            ["y"]*2 + framedata["bounding_box"]["height"]*2)

    # build child attributes for the box to hold all of the other metadata.
    for field_name in MBARI_Data.attribute_types.keys():
        attribute_element = ET.SubElement(
            box, 'attribute', attrib={'name': field_name})
        attribute_element.text = str(framedata[field_name])

    return box


def convert_annotations_to_XML(uuid_dict: {}, total_frames: int,
                               frame_offset: int) -> str:
    """ Converts a dictionary of frame data to a CVAT-readable XML.

    Parameters
    ----------
    uuid_dict : {}
      a dictionary mapping uuid's to the VisualEvent data, in the format
      {uuid: [uuid_frame1, uuid_frame2, ...], ...}
    total_frames : int
      the number of frames that were annotated.
    frame_offset: int
      the offset of the first frame of the sequence. If the first frame 
      of the sequence is labelled 24, the offset should be 24.

    Returns
    -------
    str
      a formatted XML string that can be read by CVAT as annotation data.
      Note that the class name will be used as the label in the final XML.
    """
    # Build the default structure of the XML document.
    annotations = ET.Element('annotations')
    build_sub_element(annotations, 'version', '1.1')
    meta = ET.SubElement(annotations, 'meta')

    # Build the track boxes for each uuid.
    id = 0  # an integer id for each tracked object in sequence.
    labels = {'other'}  # the labels to include, given as the class names.

    for framedata_array in uuid_dict.values():
        # Skip the framedata if the array is empty.
        if (len(framedata_array) == 0):
            continue

        # Set up the tracking Element
        track = ET.SubElement(annotations, 'track')
        track.set("id", str(id))
        id += 1
        # Set a default object label. Change this if we decide to use other labels
        track.set('label', 'Obj')
        labels.add('Obj')

        # Build the bounding box for each frame where this object was visible.
        # Sort the array so we can mark disappearances.
        framedata_array = sorted(
            framedata_array, key=lambda data: data['frame_num'])

        last_frame_num = (framedata_array[0])['frame_num'] - 1
        for i in range(0, len(framedata_array)):
            curr_framedata = framedata_array[i]
            # Check if we skipped a frame.
            if last_frame_num < curr_framedata['frame_num'] - 1:
                # Generate a dummy frame with the outside flag checked based on the
                # previous framedata.
                build_box_from_framedata(track, framedata_array[i-1],
                                         frame_override=last_frame_num - frame_offset + 1,
                                         outside=1)
            # Build the bounding box for this frame.
            build_box_from_framedata(track, curr_framedata,
                                     frame_override=curr_framedata['frame_num'] - frame_offset)
            last_frame_num = curr_framedata['frame_num']

        # Generate a dummy frame at the end of the sequence, if we haven't reached
        # the end of the video.
        final_frame = framedata_array[len(framedata_array) - 1]
        if (final_frame['frame_num'] - frame_offset < total_frames - 1):
            build_box_from_framedata(track, framedata_array[len(framedata_array) - 1],
                                     frame_override=final_frame['frame_num'] -
                                     frame_offset + 1,
                                     outside=1)

    # Build rest of the metadata for the XML document.
    # I'm actually omitting a lot of it because I'm assuming OpenCV will work??
    task = build_sub_element(meta, 'task')

    #build_sub_element(task, 'created', datetime.isoformat(datetime.now()))
    build_sub_element(task, 'mode', 'interpolation')
    build_sub_element(task, 'flipped', 'False')
    build_sub_element(task, 'overlap', '0')
    build_sub_element(task, 'start_frame', '0')
    build_sub_element(task, 'stop_frame', str(total_frames - 1))
    build_sub_element(task, 'size', str(total_frames))

    segments = build_sub_element(task, 'segments')
    segment = build_sub_element(segments, 'segment')
    build_sub_element(segment, 'start', '0')
    build_sub_element(segment, 'stop', str(total_frames - 1))

    # Build metadata for each of the label names.
    labels_xml = build_sub_element(task, 'labels')
    for label_name in labels:
        curr_label = build_sub_element(labels_xml, 'label')
        build_sub_element(curr_label, 'name', label_name)

        # Each label name has a series of attributes, each of which must include
        # data about the type, mutability, and name.
        label_attributes = build_sub_element(curr_label, 'attributes')
        for attribute_name in MBARI_Data.attribute_types.keys():
            curr_attribute = build_sub_element(label_attributes, 'attribute')
            build_sub_element(curr_attribute, 'name', attribute_name)
            build_sub_element(curr_attribute, 'mutable',
                              MBARI_Data.attribute_mutability[attribute_name])
            build_sub_element(curr_attribute, 'input_type',
                              MBARI_Data.attribute_types[attribute_name])
            build_sub_element(curr_attribute, 'default_value')
            build_sub_element(curr_attribute, 'values')
    return ET.tostring(annotations)


if __name__ == "__main__":
    """ Converts the specified MBARI JSON frame data to a CVAT-compatible annotation file.

    Usage
    -----
    python convert_json.py [output xml file] [*json files]
    """
    if (len(sys.argv) < 3):
        print("Missing one or more arguments.")
        print("Usage: [output xml file] [*json files]")
        sys.exit()

    print("Starting...")

    # Append all matching files in the arguments to our file list.
    files = []
    for i in range(2, len(sys.argv)):
        files.extend(glob.glob(sys.argv[i]))
    print("Found {} JSON frames.".format(len(files)))

    # Read the file annotations to XML.
    framedata = read_JSON_annotations(files)

    # Get the frame offset. This should be the smallest frame number in the sequence.
    # Because framedata is a dictionary where {uuid: [frame1, frame2, ...], ...}, we
    # extract the minimum frame_num from each frame.
    frame_offset = float("inf")
    for uuid in framedata.keys():
        for frame in framedata[uuid]:
            frame_offset = min(frame_offset, frame['frame_num'])

    # Note that we assume each file to correspond to exactly one frame.
    xml_string = convert_annotations_to_XML(
        framedata, len(files), frame_offset)
    formatted_xml = xml.dom.minidom.parseString(xml_string).toprettyxml()

    # Output to the file, truncating any existing data.
    file_output = open(sys.argv[1], 'w')
    file_output.truncate()
    file_output.write(formatted_xml)
    file_output.close

    print("Operation successful.")
