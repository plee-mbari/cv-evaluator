# 6/23/2020
# plee@mbari.org
# Copyright Peyton Lee, MBARI 2020

import convert_json as cj
import evaluate_model as em
import motmetrics as mm
import json
import sys
import argparse
import os.path

if __name__ == "__main__":
    """ """
    parser = argparse.ArgumentParser(description="Calculates performance metrics" + 
                                                 " for a computer vision algorithm.")
    parser.add_argument('model_directory', type=str,
                         help="The directory of the model output to test against.")
    parser.add_argument('-d', '--dict', help="The dictionary JSON file to use for classifications."
                        + " Otherwise, searches for a 'dictionary.json' file in the local path.",
                        default="dictionary.json")
    parser.add_argument('-c', '--config', help="The config JSON file to use for testing."
                        + " Otherwise, searches for a 'config.json' file in the local path.",
                        default="config.json")
    parser.add_argument('-t', '--test_directory', help="The test directory to use as truth data."
                        + " By default, uses './test/' in the local path.",
                        default="./test/")

    args = parser.parse_args()

    # Try loading the phylogeny dictionary.
    phylogenydict = {}
    with open(args.dict) as d:
        phylogenydict = json.load(d)
    pc = em.PhylogenyComparator(dictionary=phylogenydict)

    # Load the config file and the tests.
    config = {}
    with open(args.config) as c:
        config = json.load(c)
    tests = config["tests"]
    if not tests:
        print("No tests found. (Register tests in config.json)")
        exit()
    
    # For each test, verify that we have a matching .xml file in our test directory.
    for test in tests:
        fpath = "{DIR}/{TEST}.xml".format(DIR=args.test_directory, TEST=test)
        if not os.path.exists(fpath):
            print("Could not find {TEST}.xml in {DIR}".format(TEST=test, DIR=args.test_directory))
            exit()

    print("{} tests loaded. Starting...".format(len(tests)))

    # For each test, determine whether the model directory has the necessary xml file. If not,
    # run the JSON file converter. Then, evaluate and add it to our accumulator.
    results_acc = []
    results_meta = []
    for testname in tests:
        model_dir = "{DIR}/{TEST}".format(DIR=args.model_directory, TEST=testname)
        if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
            print("Could not find model directory {DIR}.".format(DIR=model_dir))
            exit()
        
        truth_file = "{DIR}/{TEST}.xml".format(DIR=args.test_directory, TEST=testname)
        track_file = "{MODEL_DIR}/result.xml".format(MODEL_DIR=model_dir)

        if not os.path.exists(track_file):
            # No result XML file was found, so we convert the JSON data in the test file.
            files = cj.get_filepaths("{MODEL_DIR}/f*.json".format(MODEL_DIR=model_dir))
            print("No result.xml found for test {TEST}. Converting {FNUM} JSON files to XML.".format(
                TEST=test, FNUM=len(files)))
            cj.convert_json_to_xml(track_file, files)
        
        # Evaluate the tracker against the truth file and save the results.
        truth_framedata = em.parse_XML_by_frame(truth_file)
        track_framedata = em.parse_XML_by_frame(track_file)
        mot_acc = em.build_mot_accumulator(truth_framedata, track_framedata)
        meta_table = em.build_metadata_table(truth_framedata, track_framedata, mot_acc, pc)

        # Save the results.
        results_acc.append(mot_acc)
        results_meta.append(meta_table)
    
    print(results_acc)
    # Print the results. Down the line we'll probably want to save the metadata tables to
    # an Excel spreadsheet.
    mh = mm.metrics.create()
    summary = mh.compute_many(results_acc, names=tests, generate_overall=True)
    strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)

