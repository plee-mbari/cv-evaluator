#!/usr/bin/env python
__author__ = 'Peyton Lee'
__copyright__ = '2020'
__license__ = 'GPL v3'
__contact__ = 'plee at mbari.org'
__doc__ = '''
Reads and converts JSON computer vision tracker output to a CVAT-compatible XML format.

@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: development
@license: GPL
'''

import glob
import json
import sys
import xml.dom.minidom
import xml.etree.ElementTree as ET

OUTPUT_ATTRIBUTE_TYPES = {'class_index': 'number',
<<<<<<< HEAD
                          'class_name': 'text',
                          'confidence': 'number',
                          'occluded_pixels': 'number',
                          'surprise': 'number',
                          'uuid': 'text'}

OUTPUT_ATTRIBUTE_MUTABILITY = {'class_index': 'False',
                               'class_name': 'False',
                               'confidence': 'True',
                               'occluded_pixels': 'True',
                               'surprise': 'True',
                               'uuid': 'False'}

=======
                       'class_name': 'text',
                       'confidence': 'number',
                       'occlusion': 'number',
                       'surprise': 'number',
                       'uuid': 'text'}

OUTPUT_ATTRIBUTE_MUTABILITY = {'class_index': 'False',
                            'class_name': 'False',
                            'confidence': 'True',
                            'occlusion': 'True',
                            'surprise': 'True',
                            'uuid': 'False'}
>>>>>>> 24bf1cc59795f76d291c366506caaa17f88bb02e

def read_json_annotations(json_file_paths) -> {}:
    """ Reads the set of JSON annotations into a map from the uuid's to the Visual
    Events.

    Parameters
    ----------
    json_file_paths : an array of JSON file paths to read from. These should be from the same
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
    for json_file in json_file_paths:
        # Read in the file JSON
        with open(json_file) as f:
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


def build_sub_element(parent: ET.Element, name: str, text="",  attrib={}) -> ET.Element:
    """ Builds an XML SubElement with the given parent, name, and text.
    """
    ret = ET.SubElement(parent, name, attrib=attrib)
    ret.text = text
    return ret


def build_box_from_framedata(parent_element: ET.Element,
                             framedata: {}, compression_ratio: float, frame_override=-1,
                             outside=0, occluded=0, keyframe=1):
    """ Creates and returns a box XML element with coordinates and attributes
        given by the framedata.

        Parameters
        ----------
        parent_element: The parent element for the XML box.

        framedata : The dictionary data for the given frame, as imported from MBARI standard
          data.

        compression_ratio : float
          (optional) The compression ratio of the image. All coordinates in the detection
          are divided by the compression ratio to get the final corners of the bounding
          box. Default is 0.5.

        Kwargs
        ------
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
    if frame_override == -1:
        box.attrib["frame"] = str(framedata["frame_num"] - 1)
    else:
        box.attrib["frame"] = str(frame_override)
    # Set the three flag values.
    box.attrib["outside"] = str(outside)
    box.attrib["occluded"] = str(occluded)
    box.attrib["keyframe"] = str(keyframe)

    # Top left x, y and bottom right x, y of the bounding box.
    box.attrib["xtl"] = str(framedata["bounding_box"]["x"] / compression_ratio)
    box.attrib["ytl"] = str(framedata["bounding_box"]["y"] / compression_ratio)
    box.attrib["xbr"] = str(framedata["bounding_box"]["x"] / compression_ratio +
                            framedata["bounding_box"]["width"] / compression_ratio)
    box.attrib["ybr"] = str(framedata["bounding_box"]["y"] / compression_ratio +
                            framedata["bounding_box"]["height"] / compression_ratio)

    # build child attributes for the box to hold all of the other metadata.
    for field_name in OUTPUT_ATTRIBUTE_TYPES.keys():
        attribute_element = ET.SubElement(
            box, 'attribute', attrib={'name': field_name})
        attribute_element.text = str(framedata[field_name])

    return box


def convert_annotations_to_XML(uuid_dict: {}, total_frames: int,
                               frame_offset: int, compression_ratio) -> str:
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

    compression_ratio : float
      the compression ratio used on the tracker.

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
    det_id = 0  # an integer id for each tracked object in sequence.
    labels = {'other'}  # the labels to include, given as the class names.

    for framedata_array in uuid_dict.values():
        # Skip the framedata if the array is empty.
        if (len(framedata_array) == 0):
            continue

        # Set up the tracking Element
        track = ET.SubElement(annotations, 'track')
        track.set("id", str(det_id))
        det_id += 1
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
                build_box_from_framedata(track, framedata_array[i-1], compression_ratio,
                                         frame_override=last_frame_num - frame_offset + 1,
                                         outside=1)
            # Build the bounding box for this frame.
            build_box_from_framedata(track, curr_framedata, compression_ratio,
                                     frame_override=curr_framedata['frame_num'] - frame_offset)
            last_frame_num = curr_framedata['frame_num']

        # Generate a dummy frame at the end of the sequence, if we haven't reached
        # the end of the video.
        final_frame = framedata_array[len(framedata_array) - 1]
        if (final_frame['frame_num'] - frame_offset < total_frames - 1):
            build_box_from_framedata(track,
                                     framedata_array[len(framedata_array) - 1],
                                     compression_ratio,
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
        for attribute_name in OUTPUT_ATTRIBUTE_TYPES.keys():
            curr_attribute = build_sub_element(label_attributes, 'attribute')
            build_sub_element(curr_attribute, 'name', attribute_name)
            build_sub_element(curr_attribute, 'mutable',
                              OUTPUT_ATTRIBUTE_MUTABILITY[attribute_name])
            build_sub_element(curr_attribute, 'input_type',
                              OUTPUT_ATTRIBUTE_TYPES[attribute_name])
            build_sub_element(curr_attribute, 'default_value')
            build_sub_element(curr_attribute, 'values')
    return ET.tostring(annotations)


def get_filepaths(*args):
    """ Parses the given regex into a list of file paths.

        Params:
        -------
        args:
            The regular expression(s) to parse into file paths.
    """
    files = []
    for x in args:
        # Check if x is a list or a file and handle separately.
        if isinstance(x, list):
            for file in x:
                files.extend(glob.glob(file))
        else:
            files.extend(glob.glob(x))
    return files


def convert_json_to_xml(destination: str, json_src: [], compression_ratio: float):
    """ Converts the list of json source files to a single, CVAT-compatible
        annotation file and writes to the defined destination.
    """
    # Read the file annotations to XML.
    framedata = read_json_annotations(json_src)

    # Get the frame offset. This should be the smallest frame number in the sequence
    # of JSON frames.
    # Strip out the six-digit frame number from the format "path/f------.xml"
    min_json_file = min(json_src)
    frame_offset = int(min_json_file[-9:-5])

    # Note that we assume each file to correspond to exactly one frame.
    xml_string = convert_annotations_to_XML(
        framedata, len(json_src), frame_offset, compression_ratio)
    formatted_xml = xml.dom.minidom.parseString(xml_string).toprettyxml()

    # Output to the file, truncating any existing data.
    file_output = open(destination, 'w')
    file_output.truncate()
    file_output.write(formatted_xml)
    file_output.close


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
    files = get_filepaths(sys.argv[2: len(sys.argv)])
    print("Found {} JSON frames.".format(len(files)))

    convert_json_to_xml(sys.argv[1], files, 0.5)

    print("Operation successful.")
