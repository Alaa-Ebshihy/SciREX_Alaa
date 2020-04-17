#! /usr/bin/env python
from collections import defaultdict
import json
import os
from sys import argv
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from allennlp.common.util import import_submodules
from allennlp.data import DataIterator, DatasetReader
from allennlp.data.dataset import Batch
from allennlp.models.archival import load_archive
from allennlp.nn import util as nn_util

import logging
logging.basicConfig(format="%(asctime)s:%(levelname)s:%(message)s", level=logging.INFO)


def predict(archive_folder, test_file, output_file, cuda_device):
    import_submodules("scirex")
    logging.info("Loading Model from %s", archive_folder)
    archive_file = os.path.join(archive_folder, "model.tar.gz")
    archive = load_archive(archive_file, cuda_device)
    model = archive.model
    model.eval()

    link_threshold = json.load(open(archive_folder + '/metrics.json'))['best_validation__span_threshold']

    model.prediction_mode = True
    config = archive.config.duplicate()
    dataset_reader_params = config["dataset_reader"]
    dataset_reader = DatasetReader.from_params(dataset_reader_params)
    dataset_reader.prediction_mode = True
    instances = dataset_reader.read(test_file)

    for instance in instances :
        batch = Batch([instance])
        batch.index_instances(model.vocab)

    data_iterator = DataIterator.from_params(config["validation_iterator"])
    iterator = data_iterator(instances, num_epochs=1, shuffle=False)

    with open(output_file, "w") as f, open(test_file + '.linked_ss', "w") as g:
        documents = {}
        for batch in tqdm(iterator):
            batch = nn_util.move_to_device(batch, cuda_device)  # Put on GPU.
            output_res = model.decode_links(batch, link_threshold)

            cluster_size = output_res['clusters_size']
            metadata = output_res['linked']['metadata'][0]
            doc_id = metadata['doc_key']
            coref_key_map = {k:i for i, k in metadata['document_metadata']['cluster_name_to_id'].items()}

            linked_clusters = {coref_key_map[i]:int(c) for i, c in enumerate(cluster_size)}
            if doc_id not in documents :
                documents[doc_id] = {}
                documents[doc_id]['linked_clusters'] = {k:0 for k in metadata['document_metadata']['cluster_name_to_id']}
                documents[doc_id]['doc_id'] = doc_id
                

            for k, c in linked_clusters.items() :
                documents[doc_id]['linked_clusters'][k] += c

        original_documents = [json.loads(line) for line in open(test_file)]
        for d in original_documents :
            assert d['doc_id'] in documents
            try :
                d['coref'] = {k:v for k, v in d['coref'].items() if documents[d['doc_id']]['linked_clusters'][k] > 0}
            except :
                breakpoint()

        f.write("\n".join([json.dumps(x) for x in documents.values()]))
        g.write("\n".join([json.dumps(x) for x in original_documents]))


def main():
    archive_folder = argv[1]
    test_file = argv[2]
    output_file = argv[3]
    cuda_device = int(argv[4])
    predict(archive_folder, test_file, output_file, cuda_device)


if __name__ == "__main__":
    main()
