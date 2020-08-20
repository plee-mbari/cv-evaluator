#!/usr/bin/env python
__author__ = 'Peyton Lee'
__copyright__ = '2020'
__license__ = 'GPL v3'
__contact__ = 'plee at mbari.org'
__doc__ = '''
Evalutes a computer vision tracker by comparing its output to
a ground-truth dataset, generating CLEAR MOT metrics.

@var __date__: Date of last svn commit
@undocumented: __doc__ parser
@status: development
@license: GPL
'''

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from json import JSONDecodeError

import motmetrics as mm
import numpy as np
import pandas as pd
import requests


class PhylogenyComparator:
    """ Gets and computes the phylogenetic differences between classifications,
        using the MBARI Knowledgebase server. Stores a dictionary used to
        replace classifications.
    """
    # Stores any previously made requests, where the key is the name and the value
    # is the complete JSON result.
    _phylogeny_request_cache = {}

    # Stores the lists of any previous phylogeny results, where the keys are tuples
    # made of (name, rank_limit) and the values are the ordered phylogeny.
    _phylogeny_cache = {}

    # Stores the results of any previous comparisons between two phylogenies, where
    # the keys are (name1, name2) and the values are the number of differing
    # classifications.
    _phylogeny_comparison_cache = {}

    # Custom dictionary. Used to override the formatted form of a classification,
    # declared at initialization.
    _dictionary = {}

    def __init__(self, dictionary={}):
        """ Constructs a new PhylogenyComparator. If a dictionary is set, uses the
            provided dictionary to override classifications when querying the server.
        """
        self._dictionary = dictionary

    def __find_detection(self, src_framedata: [], frame: int, id: int):
        """ Returns the object in the framedata dictionary that has the matching frame
            and id.

            Params
            ------
            src_framedata: {}
            A dictionary where the keys are frame numbers and the values are lists
            of detections, formatted as dictionary objects.
            frame: int
            The frame number to match.
            id: int
            The tracking id number to match.

            Returns
            -------
            The dictionary object with the matching frame number and id.
            Otherwise, returns an empty dictionary.
        """
        detections = src_framedata.get(str(frame), [])
        results = list(
            filter(lambda x: 'id' in x and int(x['id']) == id, detections))
        if results:
            return results[0]
        return {}

    def get_phylogeny(self, name: str, format_name=True, rank_limit=None) -> [str]:
        """Queries the Knowledgebase server and returns the phylogenetic ancestors
        of a given taxonomic concept as a string list.

        Parameters
        ----------
            name: str
            The name to query the database for.

            format_name: boolean
            (optional) whether the name should be automatically formatted. If there
            is a dictionary entry for that name, replaces the name with the dictionary
            entry. Otherwise, the first character is capitalized and all '_' are replaced with '%20'.

            rank_limit: str
            (optional) if set, limits the results to only classifications at or
            below that rank (ex: phylum, kingdom, superkingdom)

        Returns
        -------
            [str]
            The ordered classification layers, in descending classification rank. 
        """
        # Copy the name so we can make formatting changes to it.
        name_copy = name
        # Format the name so only the first character is capitalized.
        if format_name:
            if name in self._dictionary:
                name_copy = self._dictionary[name_copy]
            else:
                name_copy = name.capitalize()
                name_copy = name_copy.replace('_', '%20')

        # Return a copy of any previously found results.
        if (name_copy, rank_limit) in self._phylogeny_cache:
            return list(self._phylogeny_cache[(name_copy, rank_limit)])

        json_result = {}
        if name_copy not in self._phylogeny_request_cache:
            # Make the server request and parse to json.
            req_result = requests.get(
                "http://dsg.mbari.org/kb/v1/phylogeny/up/" + name_copy)
            try:
                json_result = json.loads(req_result.text)
                self._phylogeny_request_cache[name_copy] = json_result
            except JSONDecodeError:
                print("Could not get Knowledgebase phylogeny for {}".format(name_copy))
                self._phylogeny_request_cache[name_copy] = {}
                return []
        else:
            json_result = self._phylogeny_request_cache[name_copy]

        # Loop through the nested JSON and get the names.
        ret = []
        curr_level = json_result
        while True:
            # Limit results to only at or below the rank_limit by clearing any parents.
            if 'rank' in curr_level and curr_level['rank'] == rank_limit:
                ret.clear()
            # Append the category name to our results
            if 'name' in curr_level:
                ret.append(curr_level['name'])
            # Recurse on children if available, otherwise exit.
            if 'children' in curr_level and curr_level['children']:
                curr_level = curr_level['children'][0]
            else:
                break
        return ret

    def compare_phylogeny(self, truth_name: str, track_name: str, format_name=False,
                          rank_limit=None) -> int:
        """Returns the number of differing phylogenetic classifications between the 
        truth_name and track_name, as well as the largest number of phylogenetic
        classification layers.

        Params
        ------
        truth_name: str
            The classification of an object given by the ground-truth data.

        track_name: str
            The classification of an object given by the model.

        format_name: bool
            (optional) If true, automatically formats the provided names. False by default.

        rank_limit: str
            (optional) The maximum phylogenetic ranking that should be considered. If set,
            the maximum number of phylogenetic classifications is set at the rank_limit,
            if found.

        Returns
        -------
        (int, int)
        1) The differences between the two phylogenetic sequences and 2) the max
        number of classification layers present.
        """
        if truth_name is None or track_name is None:
            return (0, 0)

        # Check if we've already made this comparison before. If so, we can return
        # our cached results.
        if (truth_name, track_name) in self._phylogeny_comparison_cache:
            return self._phylogeny_comparison_cache[(truth_name, track_name)]
        if (track_name, truth_name) in self._phylogeny_comparison_cache:
            return self._phylogeny_comparison_cache[(track_name, truth_name)]

        # Get the taxonomy sequence for both truth and tracked classifications.
        truth_phylogeny = self.get_phylogeny(
            truth_name, format_name, rank_limit)
        track_phylogeny = self.get_phylogeny(
            track_name, format_name, rank_limit)

        differences = 0
        # Count any differences in the sequences
        for i in range(0, min(len(truth_phylogeny), len(track_phylogeny))):
            if truth_phylogeny[i] != track_phylogeny[i]:
                differences += 1
        # Count any length differences (which indicate differing specificity)
        if (len(truth_phylogeny) != len(track_phylogeny)):
            differences += abs(len(truth_phylogeny) - len(track_phylogeny))
        # Save a copy of the results.
        self._phylogeny_comparison_cache[(truth_name, track_name)] = (differences,
                                                                      max(len(truth_phylogeny), len(track_phylogeny)))
        return (differences, max(len(truth_phylogeny), len(track_phylogeny)))

    def compare_phylogeny_from_row(self, row: pd.Series, truth_data, model_data):
        """ Computes the differences in phylogeny for a single row of the
            MOTAccumulator's event table. (intended as lambda function)

            Params
            ------
            row: pandas.Series
                The row of the MOTAccumulator, as a Series. The name of the 
                row should be in the form (frame_num, event_id) and it 
                should include the columns 'OId' and 'HId'.

            truth_data: {}
                The parsed truth output. This should be in a form matching
                the output of parse_XML_by_frame.

            model_data: {}
                The parsed model output.

            Returns
            -------
            (int, int), where the tuple is made up of (differing classifications,
            total classifications) for that match.
        """
        frame_num = int(row.name[0])
        # Find the dictionary corresponding to the frame and ID
        truth_frame = self.__find_detection(truth_data, frame_num, row['OId'])
        model_frame = self.__find_detection(model_data, frame_num, row['HId'])

        # Check if we found the detections, if not we return.
        if not model_frame or not truth_frame:
            return (0, 0)

        # Compare the phylogeny of the two
        diff, total = self.compare_phylogeny(truth_frame['class_name'],
                                             model_frame['class_name'],
                                             format_name=True,
                                             rank_limit='kingdom')
        return (diff, total)

    def add_definition(self, key: str, value: str):
        """Adds a new definition to the internal dictionary used to replace classifications."""
        self._dictionary[key] = value

# End class PhylogenyComparator


def parse_XML_by_frame(xml_file: str, name_dictionary={}):
    """ Parses a CVAT output XML string into a dictionary, where the keys are the
        frame numbers and the corresponding values are arrays of the attributes of
        the detections.
        An optional dictionary can be passed in, which will be used to replace the 
        class_name property if a match is found.

        ex:

        {0: [{detection1}, {detection2}, ...,],
         1: [],
         2: [{detection3}, {detection4}, ...,]}

        A detection consists of a dictionary of all attributes in the detection
        (track and box attributes).

        Frames marked outside are ignored.
    """
    ret = {}
    # Parse the XML Tree for traversal.
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for track in root.findall('track'):
        # Traverse all bounding boxes in this track.
        for box in track.findall('box'):
            # Skip any bounding boxes that are marked as outside (end of track)
            if box.get('outside', 0) == 1:
                continue
            # Copy all of the frame's attributes to a dictionary object.
            frame_dict = {}
            frame_dict.update(track.attrib)
            frame_dict.update(box.attrib)

            # Get all of the attribute subelements.
            for attribute in box.findall('attribute'):
                frame_dict[attribute.get('name')] = attribute.text

            # Check if the class_name property is in our dictionary, and
            # if so, replace it.
            if frame_dict['class_name'] in name_dictionary:
                frame_dict['class_name'] = name_dictionary[frame_dict['class_name']]

            # Save the frame_dict.
            frame_num = box.get('frame')
            if frame_num not in ret:
                ret[frame_num] = []
            ret[frame_num].append(frame_dict)
    return ret


def build_cost_matrix(truth_objects: [], model_objects: [], iou=False,
                      iou_cutoff=1., d2_cutoff=1000.):
    """ Builds a cost matrix between the detection objects of a truth
        and model dataset.

        Params
        ------
        truth_objects: [{}]
          A list of truth detections, where each detection is an object 
          with the keys xbr, xtl, ybr, xtl corresponding to number coords.

        model_objects: [{}]
          A list of model output detections, in the same format.

        iou: boolean
          (optional) if true, uses intersection over union instead of
          euclidean norm squared (distance between centers, default).

        d2_cutoff: float
          (optional) the cutoff for the euclidean norm squared distance calculation.

        iou_cutoff: float
          (optional) if iou is true, determines the cutoff for the intersection
          over union.

        Returns
        -------
          A numpy 2D array, where the cell[i][j] represents the cost
          between truth_objects[i] and model_objects[j].
    """
    # Build and populate the matrix with the distances
    if iou:
        # Use Intersection over Union
        # Convert detection boxes to coords given by [X, Y, Width, Height]
        truth_coords = list(map(lambda x: [float(x['xtl']),
                                           float(x['ytl']),
                                           float(x['xbr']) - float(x['xtl']),
                                           float(x['ybr']) - float(x['ytl'])],
                                truth_objects))
        model_coords = list(map(lambda x: [float(x['xtl']),
                                           float(x['ytl']),
                                           float(x['xbr']) - float(x['xtl']),
                                           float(x['ybr']) - float(x['ytl'])],
                                model_objects))
        return mm.distances.iou_matrix(np.array(truth_coords),
                                       np.array(model_coords),
                                       max_iou=iou_cutoff)
    else:
        # Use Euclidean Norm2Squared
        # Convert detection boxes to center points, [X, Y]
        truth_coords = np.array(list(map(lambda x: [(float(x['xbr']) - float(x['xtl']))/2.,
                                                    (float(x['ybr']) - float(x['ytl']))/2.],
                                         truth_objects)))
        model_coords = np.array(list(map(lambda x: [(float(x['xbr']) - float(x['xtl']))/2.,
                                                    (float(x['ybr']) - float(x['ytl']))/2.],
                                         model_objects)))
        matrix = mm.distances.norm2squared_matrix(np.array(truth_coords),
                                                  np.array(model_coords),
                                                  max_d2=d2_cutoff)
        # Normalize to cutoff
        npmatrix = np.array(matrix)
        return npmatrix / d2_cutoff


def build_mot_accumulator(truth_framedata: {},
                          model_framedata: {}) -> mm.MOTAccumulator:
    """ Parses the provided frame data and builds the motmetrics accumulator with
        it.

        Params
        ------
        truth_framedata: The data for the ground-truth output, parsed as a dictionary
              where the keys are frame numbers mapping to lists of detections for that
              frame. (as given by parse_XML_by_frame)

        model_framedata: The data for the model's output, in the same format as the
              truth_framedata.

        Returns 
        -------
        A motmetrics.MOTAccumulator that stores the matches and MOTEvents for all the
        frames in the input framedata.
    """

    # Get the maximum number of frames.
    max_frame = -1
    if truth_framedata:
        truth_frames = list(map(lambda x: int(x), truth_framedata.keys()))
        max_frame = max(truth_frames)
    if model_framedata:
        model_frames = list(map(lambda x: int(x), model_framedata.keys()))
        max_model_frame = max(model_frames)
        max_frame = max(max_frame, max_model_frame)

    #print("max_frame is {}".format(max_frame))

    acc = mm.MOTAccumulator()
    # Loop through each frame and build the distance matrix.
    for i in range(max_frame + 1):
        # Get the array of detections for the current frame
        truth_curr_frame = truth_framedata.get(str(i), [])
        model_curr_frame = model_framedata.get(str(i), [])

        # Build the distance/cost matrix.
        dist_matrix = build_cost_matrix(
            truth_curr_frame, model_curr_frame, iou=True)

        # Get the object indices of detections for the truth and model as arrays.
        truth_indices = list(map(lambda det: int(det['id']), truth_curr_frame))
        model_indices = list(map(lambda det: int(det['id']), model_curr_frame))

        # Pass into the motmetrics accumulator, which will automatically match
        # the indices.
        acc.update(truth_indices, model_indices, dist_matrix, frameid=i)
    return acc


def find_detection(src_framedata: [], frame: int, id: int):
    """ Returns the object in the framedata dictionary that has the matching frame
        and id.

        Params
        ------
        src_framedata: {}
          A dictionary where the keys are frame numbers and the values are lists
          of detections, formatted as dictionary objects.

        frame: int
          The frame number to match.

        id: int
          The tracking id number to match.

        Returns
        -------
          The dictionary object with the matching frame number and id.
          Otherwise, returns an empty dictionary.
    """
    detections = src_framedata.get(str(frame), [])
    results = list(
        filter(lambda x: 'id' in x and int(x['id']) == id, detections))
    if results:
        return results[0]
    return {}


def build_metadata_table(truth_framedata: {}, model_framedata: {},
                         mot_acc: mm.MOTAccumulator,
                         phylogeny_comparator=PhylogenyComparator()) -> pd.DataFrame:
    """ Builds a DataFrame that extends the accumulator's event table with  
        metadata and phylogenetic comparisons from the model's framedata.

        This allows one to quantify the performance of the model in relation to
        the metadata tags.

        Params
        ------
        truth_framedata: The truth output, parsed as a dictionary where the keys 
            are frame numbers and the values are lists of detection data, as 
            given by parse_XML_by_frame.

        model_framedata: The model output, matching the format specified for
            truth_framedata.

        mot_acc: A motmetrics.MOTAccumulator to retrieve the MOTEvents from, as
            given by build_mot_accumulator.

        phylogeny_comparator: (optional) A PhylogenyComparator to be used for
            calculating the differences between calculations. (Reusing the same
            PhylogenyComparator for multiple evaluations is more efficient!)

        Returns
        -------
        A copy of the mot_acc events DataFrame with added columns for the detection 
            metadata (such as confidence, surprise, class_name, etc.) and the 
            differences in phylogenetic classification.
        """
    # The mot_events are a DataFrame with columns [FrameId, Event, Type, OId, HId,
    # D] => (D being distance).
    events = mot_acc.mot_events.copy()

    # The meta tags that we will store in the new DataFrame.
    meta_tags = ["confidence", "occluded_pixels", "surprise"]
    # Indicates meta tags that should be casted to numbers.
    meta_num = [True, True, True]

    for i in range(len(meta_tags)):
        # Get the corresponding detection to this row, and add in a new column
        # for that meta tag.
        tag = meta_tags[i]
        is_num = meta_num[i]
        if is_num:
            events[tag] = events.apply(
                lambda x: pd.to_numeric(find_detection(model_framedata,
                                                       # Frame number
                                                       x.name[0],
                                                       x['HId'])   # The ID number of the detection
                                        .get(tag, float('nan'))),  # Get the tag property.
                axis=1)
        else:
            events[tag] = events.apply(
                lambda x: find_detection(model_framedata,
                                         x.name[0],  # Frame number
                                         x['HId'])   # The ID number of the detection
                .get(tag, float('nan')),  # Get the tag property.
                axis=1)
    # Add in the classification names for both the truth and model detection.
    events['Oclass_name'] = events.apply(
        lambda x: find_detection(truth_framedata, x.name[0], x['OId']).get(
            'class_name', "N/A"),
        axis=1
    )
    events['Hclass_name'] = events.apply(
        lambda x: find_detection(model_framedata, x.name[0], x['HId']).get(
            'class_name', "N/A"),
        axis=1
    )

    # Add in the phylogenetic classifications, one column for the totals and one
    # for the differences.
    events['classif_diff'] = events.apply(
        lambda x: phylogeny_comparator.compare_phylogeny_from_row(x,
                                                                  truth_framedata, model_framedata)[0],
        axis=1)
    events['classif_total'] = events.apply(
        lambda x: phylogeny_comparator.compare_phylogeny_from_row(x,
                                                                  truth_framedata, model_framedata)[1],
        axis=1)

    return events


def get_species_breakdown(metadata_tables: [pd.DataFrame]) -> pd.DataFrame:
    """ Creates a dataframe with metrics on the detection of each species.

        Params
        ------
        metadata_tables: A list of metadata DataFrames to extract species from.

        Returns
        -------
        A DataFrame that shows the number of occurrences for each event (MATCH, SWITCH, MISS, etc.)
        by species labels.
    """
    # Generate a set of all classifications that were made.
    classifiers = []
    for table in metadata_tables:
        # Concatenate the observation and hypothesis names, then filter so only unique labels are left.
        classifiers = np.concatenate(
            (classifiers, table["Oclass_name"].unique(), table["Hclass_name"].unique()))
        # Filter out None and N/A, then simplify to unique values.
        classifiers = np.unique(
            classifiers[(classifiers != np.array(None)) & (classifiers != np.array("N/A"))])

    # Columns include counts for each of the events, plus all other classifiers as guesses.
    columns = ["MATCH", "SWITCH", "MISS", "FP",
               "TRANSFER", "ASCEND", "MIGRATE", "OFreq", "HFreq"]
    columns.extend(classifiers)

    ret_df = pd.DataFrame(columns=columns, index=classifiers)

    # Merge all of the metadata tables into one supertable.
    meta_df = pd.DataFrame()
    for table in metadata_tables:
        meta_df = meta_df.append(table)

    for class_name in classifiers:
        # Find all rows of the supertable that correspond to the class_name
        rows = meta_df.query(
            'Oclass_name=="{NAME}" | Hclass_name=="{NAME}"'.format(NAME=class_name))
        # Generate a series and count the number of matching events.
        """
        for event in accepted_events:
            event_rows = rows.query('Type=="{}"'.format(event))
            class_series[event] = len(event_rows)
        """
        event_count = rows['Type'].value_counts()

        # For each classifier, count the number of matches that were made with this class_name and add
        # to our series.
        matches = rows.query('Type=="MATCH"')
        hclass_count = matches['Hclass_name'].value_counts()
        oclass_count = matches['Oclass_name'].value_counts()

        freq = {}
        freq["HFreq"] = hclass_count.get(class_name, 0)
        freq["OFreq"] = oclass_count.get(class_name, 0)
        freq_count = pd.Series(freq)

        tclass_count = hclass_count.add(oclass_count, fill_value=0)

        # Because this class is double-counted, we subtract the number of matches from
        # the class count.
        # tclass_count[class_name] -= matches.shape[0]

        # Append the series to our ret_df.
        class_series = freq_count.append(event_count)
        class_series = class_series.append(tclass_count)
        class_series.name = class_name
        ret_df.loc[class_name] = class_series
    return ret_df


def print_evaluation(truth_file, tracker_file):
    """ Compares to XML annotation files, generating and printing evaluation metrics.
        Metrics are specified by the CLEAR MOT metrics (see https://github.com/cheind/py-motmetrics).

        Params
        ------
        truth_file: The truth XML annotation file.
        tracker_file: The tracker output XML annotation file.
    """
    truth_framedata = parse_XML_by_frame(truth_file)
    model_framedata = parse_XML_by_frame(tracker_file)

    mot_acc = build_mot_accumulator(truth_framedata, model_framedata)

    # pd.set_option('display.max_columns', None) # Previously set for debugging
    md = build_metadata_table(truth_framedata, model_framedata, mot_acc)
    classif_diff = md.loc[(md['Type'] == 'MATCH'), 'classif_diff'].sum()
    classif_total = md.loc[(md['Type'] == 'MATCH'), 'classif_total'].sum()

    mh = mm.metrics.create()
    summary = mh.compute(mot_acc, metrics=mm.metrics.motchallenge_metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    print("Classification accuracy: {:.2f}%".format(
        (1.-(classif_diff/classif_total))*100.))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalutes a computer vision tracker by comparing its output to \
                                     a ground-truth dataset, generating CLEAR MOT metrics.")
    parser.add_argument(
        'truth_xml_path', help="Path to truth XML annotation file.")
    parser.add_argument('tracker_xml_path', help="Path to tracker output XML annotation file. If using deepsea-track, output should be \
                        converted from JSON using convert_json.py.")
    args = parser.parse_args()

    print_evaluation(args.truth_xml_path, args.tracker_xml_path)
