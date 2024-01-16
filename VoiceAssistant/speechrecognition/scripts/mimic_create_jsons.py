"""Utility script to create the training and test json files for speechrecognition."""

import os
import argparse
import json
import random

def main(args):
    """
    Generates a train and test JSON file from a given text file.
    
    Args:
        args (namespace): The command-line arguments passed to the function.
            - file_folder_directory (str): The directory of the text file.
            - percent (float): The percentage of the data to be included in the train set.
            - save_json_path (str): The directory where the train and test JSON files will be saved.
    
    Returns:
        None
    """
    data = []
    directory = args.file_folder_directory
    filetxtname = args.file_folder_directory.rpartition('/')[2]
    percent = args.percent
    
    with open(directory + "/" + filetxtname + "-metadata.txt", encoding="utf-8") as f: 
        for line in f: 
            file_name = line.partition('|')[0]
            text = line.split('|')[1] 
            data.append({
            "key": directory + "/" + file_name,
            "text": text
            })

    random.shuffle(data)
    
    f = open(args.save_json_path +"/"+ "train.json", "w")
    
    with open(args.save_json_path +"/"+ 'train.json','w') as f:
        d = len(data)
        i=0
        while(i<int(d-d/percent)):
            r=data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i+1
    
    f = open(args.save_json_path +"/"+ "test.json", "w")

    with open(args.save_json_path +"/"+ 'test.json','w') as f:
        d = len(data)
        i=int(d-d/percent)
        while(i<d):
            r=data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i+1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Utility script to create the training and test json files for speechrecognition. """
    )
    parser.add_argument('--file_folder_directory', type=str, default=None, required=True,
                        help='directorio de clips proporcionados por mimic-recording-studio')
    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='Ruta al directorio donde se supone que deben guardarse los archivos JSON.')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='Porcentaje de clips colocados en test.json en lugar de train.json.')

    args = parser.parse_args()

    main(args)

