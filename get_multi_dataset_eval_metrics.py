"""
The script computes the VOC style AP@50 and Recall@50 for class agnostic object detector for multiple datasets. The
expected directory structure should be,
-> data
    -> voc2007
        -> Annotations
        -> Model_Name
            -> all_objects.pkl
            -> all_entities.pkl
            ...
            ...
            -> combined.pkl
    -> coco
        -> instances_val2017.json
        ...
        ...
    -> kitti
        -> Annotations
        ...
    -> kitchen
        ...
    -> cliaprt
        ...
    -> comic
        ...
    -> watercolor
        ...
    -> dota
        ...

Note that the script expect COCO annotations in standard coco format & annotations of all other datasets in VOC format.
"""

import argparse
import os
import glob
import csv
from get_eval_metrics import voc_eval


def parse_arguments():
    """
    Parse the command line arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_base_dir", required=False, default="data",
                    help="Path to the dataset base directory. The directory should follow the directory structure as "
                         "mentioned above.")
    args = vars(ap.parse_args())

    return args


def main():
    args = parse_arguments()
    base_dataset_dir = args["dataset_base_dir"]
    # Open the csv file to store the evaluation metrics
    header = ['Dataset', 'Model Name', 'Text Query', 'Avg. Boxes per Image', 'AP@50', 'Recall@50', 'Precission@50']
    output_file = open(f"{base_dataset_dir}.csv", "w+")
    writer = csv.writer(output_file)
    writer.writerow(header)  # Write the header
    print(header)
    # For each dataset
    for dataset in os.listdir(base_dataset_dir):
        if 'coco' in dataset.lower():
            annotation_path = f"{base_dataset_dir}/{dataset}/instances_val2017.json"
            ann_type = 'coco'
        else:
            annotation_path = f"{base_dataset_dir}/{dataset}/Annotations"
            ann_type = 'voc'
        # For each model
        for model in os.listdir(f"{base_dataset_dir}/{dataset}"):
            # For each query
            for query in glob.glob(f"{base_dataset_dir}/{dataset}/{model}/*.pkl"):
                dets_path = query
                # Calculate the metrics
                rec, prec, ap, avg_dets_per_image = voc_eval(dets_path, annotation_path, ann_type=ann_type)
                # Save the results
                metrics = [dataset, model, os.path.basename(query), avg_dets_per_image,
                           round(ap*100, 2), round(rec[-1]*100, 2), round(prec[-1]*100, 2)]
                writer.writerow(metrics)
                print(metrics)
    output_file.close()


if __name__ == "__main__":
    main()
