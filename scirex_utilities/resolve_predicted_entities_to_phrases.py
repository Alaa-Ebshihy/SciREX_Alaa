from argparse import ArgumentParser

import numpy as np

from scirex_utilities.io_util import *
from scirex_utilities.preprocess_util import *


# TODO: introduce another method based on cluster center
# TODO: in addition to the file for the names use the vectors (average of vector of words inside the cluster) and keep the cluster numbers
def find_cluster_representative(cluster_key, clusters_predictions, ner_prediction, method='first'):
    if method == 'first':
        return find_cluster_rep_by_first_candidate(clusters_predictions['clusters'][cluster_key],
                                                   ner_prediction['words'])
    return ""


def find_cluster_rep_by_first_candidate(cluster_info, words):
    rep_span = cluster_info[0]
    return '_'.join(words[rep_span[0]:rep_span[1]])


def generate_cluster_vectors_map(clusters, words):
    clusters_vectors_map = {}
    for cluster_key in clusters:
        print('getting vectors for cluster', cluster_key)
        clusters_vectors_map[cluster_key] = generate_cluster_vector_representation(clusters[cluster_key], words)
    return clusters_vectors_map


def generate_cluster_vector_representation(cluster_info, words):
    vectors = []
    for text_span in cluster_info:
        text = ' '.join(words[text_span[0]:text_span[1]])
        vectors.append(vectorize_text(text))
    sum_vectors = vectors[0]
    for i in range(1, len(vectors)):
        sum_vectors = sum_vectors + vectors[i]
    return sum_vectors / len(vectors)


def main(args):
    ner_predictions_path = args.ner_predictions_path
    clusters_predictions_path = args.clusters_predictions_path
    relation_predictions_path = args.relation_predictions_path
    output_path = args.output_path
    doc_id = args.paper_id
    resolution_method = args.resolution_method

    ner_prediction = read_json(ner_predictions_path)
    clusters_predictions = read_json(clusters_predictions_path)
    relation_predictions = read_json(relation_predictions_path)
    clusters_vectors_map = generate_cluster_vectors_map(clusters_predictions['clusters'], ner_prediction['words'])

    resolved_relations = []

    for relation_info in relation_predictions['predicted_relations']:
        if relation_info[2] == 0:
            continue
        resolved_relation_entities = []
        resolved_relation_vectors = []
        for cluster_key in relation_info[0]:
            entity_rep = find_cluster_representative(cluster_key, clusters_predictions, ner_prediction,
                                                     method=resolution_method)
            resolved_relation_entities.append(entity_rep)
            resolved_relation_vectors.append(clusters_vectors_map[cluster_key])
        resolved_relations.append([resolved_relation_entities, resolved_relation_vectors, relation_info[1]])
    resolved_relations.sort(key=lambda x: x[2], reverse=True)
    write_json(output_path, {'doc_id': doc_id, 'sorted_predicted_relations': resolved_relations}, indent=0)


if __name__ == '__main__':
    parser = ArgumentParser("Convert the predicted relations to true words from text")
    parser.add_argument("ner_predictions_path", help="The ner prediction.")
    parser.add_argument("clusters_predictions_path", help="The clusters prediction.")
    parser.add_argument("relation_predictions_path", help="The relation prediction.")
    parser.add_argument("output_path", help="The output json path of the result.")
    parser.add_argument("paper_id", help="The paper id, the pdf name should be the same.")
    parser.add_argument("--resolution_method", help="Method to resolve the cluster: first or center.", default='first')

    args = parser.parse_args()
    main(args)
