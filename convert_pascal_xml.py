#!/usr/bin/env python
__author__ = 'Peyton Lee'
__copyright__ = '2020'
__license__ = 'GPL v3'
__contact__ = 'plee at mbari.org'
__doc__ = '''
Reads and converts PASCAL XML annotation files to a CVAT-compatible XML format.

@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: development
@license: GPL
'''

import argparse
import glob
import convert_json as cj
import sys
import xml.dom.minidom
import xml.etree.ElementTree as ET

""" For converting between analogous PASCAL XML and CVAT XML properties.
"""
PASCAL_PROPERTIES_TO_CVAT = {
    "name": "class_name",
    "occluded": "occluded_pixels",
    "xmin": "xtl",
    "xmax": "xbr",
    "ymin": "ytl",
    "ymax": "ybr"
}

def read_xml_to_framedata(xml_src: []):
    """ Reads the XML source files into a dictionary, where keys are the image
        filenames and the values are objects with properties corresponding to
        the annotation's values.
    """
    # Assume that xml src files should be in lexicographic order
    xml_src.sort()

    ret = {}

    for i in range(len(xml_src)):
        xml_file = xml_src[i]
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Root should be an annotation tag.
        # Child tags: folder, source, filename, size, segmented, object(s)
        # Extract the filename and store as the key in our return dictionary.
        frame_image = root.find('filename').text
        frame_object = {}
        frame_object['id'] = i
        frame_object['width'] = root.find('size').find('width').text
        frame_object['height'] = root.find('size').find('height').text

        # Get the properties of every bounding box in this frame
        # and store it in the object.
        boxes = []  
        box_properties = ['name', 'pose', 'truncated', 'occluded', 'difficult', 'annotator', 'verifiedby', 'id']
        for box in root.findall('object'):
            box_object = {}
            for property_name in box_properties:
                box_object[property_name] = box.find(property_name).text
            for corner in ['xmin', 'ymin', 'xmax', 'ymax']:
                box_object[corner] = box.find('bndbox').find(corner).text
            boxes.append(box_object)
        frame_object['boxes'] = boxes
        # Add the object representing this frame to our return dictionary.
        ret[frame_image] = frame_object
    return ret          


def convert_pascal_xml_to_cvat_xml(destination: str, xml_src: [], compression_ratio: float):
    """ Converts the list of json source files to a single, CVAT-compatible
        annotation file and writes to the defined destination.
    """
    # Read the file annotations to XML.
    framedata = read_xml_to_framedata(xml_src)

    # Construct the main tree of the CVAT XML file.
    builder = ET.TreeBuilder()
    builder.start('annotations')
    
    # Add a CVAT version tag
    builder.start('version')
    builder.data('1.1')
    builder.end('version')

    # Build image tags for each frame in the sequence.
    for image_name in framedata.keys():
        frame_obj = framedata[image_name]
        image_attrs = {'id': str(frame_obj['id']), 
                     'name': image_name, 
                     'width': str(frame_obj['width']),
                     'height': str(frame_obj['height'])}
        builder.start('image', image_attrs)
        # Add the box tag for each bounding box
        for bbox in frame_obj['boxes']:
            # Construct the attributes of the bounding box,
            # converting property names to their CVAT homologs.
            bbox_attr_names = ['xmin', 'ymin', 'ymax', 'xmax']
            bbox_attrs = {"label": "Obj", "occluded": "0"}
            for attr_name in bbox_attr_names:
                cvat_name = PASCAL_PROPERTIES_TO_CVAT[attr_name]
                bbox_attrs[cvat_name] = str(float(bbox[attr_name]) / compression_ratio)
            builder.start('box', bbox_attrs)

            # Add child tags for remaining data, converting
            # the names if there is CVAT-specific name.
            bbox_properties = ['name', 'pose', 'truncated', 'occluded', 'difficult', 'annotator', 'verifiedby']
            for property_name in bbox_properties:
                if property_name not in bbox:
                    continue
                keyName = property_name
                if property_name in PASCAL_PROPERTIES_TO_CVAT:
                    keyName = PASCAL_PROPERTIES_TO_CVAT[property_name]
                # Build 'attribute' tag with the property name as an attribute
                builder.start('attribute', {'name': keyName})
                builder.data(str(bbox[property_name]))
                builder.end('attribute')  # close the attribute tag.
            # End the bounding box tag.
            builder.end('box')
        builder.end('image')  # End the image tag.
    builder.end('annotation')  # End the root annotation tag.

    xml_string = ET.tostring(builder.close())
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

    parser = argparse.ArgumentParser(
        description="Converts the JSON tracker output to a CVAT-compatible XML annotation file.")
    parser.add_argument('output_xml_path', type=str,
                        help="Path for the XML output file.")
    parser.add_argument('xml_file_paths', type=str, help="One or more JSON tracker output files.",
                        nargs="+")
    parser.add_argument('-c', '--compression_ratio', type=float, help="The compression ratio that the tracker was run at. \
                        If the footage was 1920x1080 but the tracks were at size 960x540, the ratio would be 0.5.",
                        default=1.0)
    args = parser.parse_args()

    output_xml_path = args.output_xml_path
    xml_file_paths = args.xml_file_paths

    # Append all matching files in the arguments to our file list.
    xml_files = cj.get_filepaths(xml_file_paths)
    print("Found {} XML frames.".format(len(xml_files)))

    convert_pascal_xml_to_cvat_xml(output_xml_path, xml_files, args.compression_ratio)

    print("Operation successful.")