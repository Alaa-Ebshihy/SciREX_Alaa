import math
from argparse import ArgumentParser

from scirex_utilities.io_util import *
from scirex_utilities.preprocess_util import *


def find_cluster_representative(cluster_key, clusters_predictions, ner_prediction, cluster_vector_map, method='first'):
    cluster_info = clusters_predictions['clusters'][cluster_key]
    if method == 'first':
        return find_cluster_rep_by_first_candidate(cluster_info, ner_prediction['words'])
    if method == 'center':  # cluster vector shouldn't be empty
        return find_cluster_rep_by_center_candidate(cluster_info, ner_prediction['words'],
                                                    cluster_vector_map[cluster_key])
    return ""


def find_cluster_rep_by_first_candidate(cluster_info, words):
    rep_span = cluster_info[0]
    return '_'.join(words[rep_span[0]:rep_span[1]])


def find_cluster_rep_by_center_candidate(cluster_info, words, cluster_vector):
    rep_text = ""
    similarity = 0
    for rep_span in cluster_info:
        text = ' '.join(words[rep_span[0]:rep_span[1]])
        text_vector = vectorize_text(text)
        cur_sim = cosine_similarity(text_vector, cluster_vector)
        if math.isnan(cur_sim):
            print('nan found')
            cur_sim = 0
        if similarity <= cur_sim:
            rep_text = '_'.join(words[rep_span[0]:rep_span[1]])
            similarity = cur_sim
    return rep_text


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
        vectors.append(vectorize_text(text.lower()))
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
                                                     clusters_vectors_map, method=resolution_method)
            resolved_relation_entities.append(entity_rep)
            resolved_relation_vectors.append(clusters_vectors_map[cluster_key].tolist())
        resolved_relations.append([resolved_relation_entities, resolved_relation_vectors, relation_info[1]])
    resolved_relations.sort(key=lambda x: x[2], reverse=True)
    write_json(output_path, {'doc_id': doc_id, 'sorted_predicted_relations': resolved_relations})


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
