# 6/23/2020
# plee@mbari.org
# Copyright Peyton Lee, MBARI 2020

import convert_json as cj
import evaluate_model as em
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
    
    # For each test, verify that we have a matching .xml file for it.
    for test in tests:
        fpath = "{DIR}/{TESTNAME}.xml".format(DIR=args.test_directory, TESTNAME=test)
        if not os.path.exists(fpath):
            print("Could not find {TEST}.xml in {DIR}".format(TEST=test, DIR=args.test_directory))
            exit()

    print("{} tests loaded. Starting...".format(len(tests)))

    



