# cv-evaluator
Benchmarks and evaluation tests for computer vision detection/tracking algorithms. Developed by Peyton Lee for the Monterey Bay Aquarium Research Institute (plee at mbari.org).

The evaluation is intended to work with the [CVAT](https://github.com/opencv/cvat) annotation tool, using the [py-motmetrics](https://github.com/cheind/py-motmetrics) library.

### Testing Organization
The main test driver is run_benchmark.py.
```
  Arguments:
  model_directory: the directory of the model output to test against.
  -d, --dict: Dictionary JSON file used to map output classifications to server-accepted names. Default uses ./dictionary.json
  -c, --config: Config file for the benchmark, which defines the tests to be run. Default uses ./config.json
  -t, --test_directory: Test directory for the ground-truth comparison data. Default uses ./test/
  -o, --output_path: the path to output an XLSX result document to. Default uses ./output.xlsx
```
  
A **test directory** must have a CVAT-compatible XML annotation file for each test, as specified in the config file.
```
/folder/
  |---- testname1.xml
  |---- testname2.xml
  |---- ...
```
  
The **model output directory** must have a matching directory for each config-specified test. The directory should either have a single JSON file for each model-output frame, or 'result.xml' file that summarizes the annotations. If no result.xml file is found, one will be automatically generated.
```
/folder/
  |--- /testname1/
        |--- f00000.json
        |--- f00001.json
        |--- ....
  |--- /testname2/
        |--- result.xml
```

The **output XLSX file** has a summary sheet, with the breakdown of the metrics for each test and the overall metrics, and an individual events page for each test sequence. 

### Module Breakdown
`convert_json.py` is used to convert the tracker output (a series of JSON files, one for each frame) to a single CVAT-compatible XML track file.

`evaluate_model.py` loads in the frame data from the XML track files and generates metrics for it. 

`run_benchmark.py` evaluates and aggregates metrics for a pre-defined test suite, with output written as an XLSX file.

### See Also:
[plee-mbari/deepsea-track-notebook](https://bitbucket.org/plee-mbari/deepsea-track-notebook): A project that makes use of this repository for evaluation of a video track. Several python notebooks show detailed steps for how the modules can be used to parse the tracker XML and JSON files.
