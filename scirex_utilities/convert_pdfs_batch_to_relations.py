import subprocess
from argparse import ArgumentParser

from scirex_utilities.preprocessing.add_cleaned_text_to_pwc import *


def convert_to_prediction_input(doc_id, json_sent):
    cur_section_id = -1
    words = []
    sentences = []
    sections = []
    for sent_info in json_sent:
        offset = len(words)
        if cur_section_id != sent_info['section_id']:
            cur_section_id = sent_info['section_id']
            sections.append([offset, offset])
        tokens = sent_info['sentence'].split()
        words.extend(tokens)
        sentences.append([offset, len(words)])
        sections[-1][1] = len(words)
    return {"doc_id": doc_id, "words": words, "sentences": sentences, "sections": sections}


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    cuda_device = args.cuda_device

    pdfs = [file for file in list_files_in_dir(input_dir) if file.endswith('.pdf')]

    for pdf in pdfs:
        paper_id = pdf[:-4]
        print("Processing paper", paper_id)
        print("--------------------------")
        # 1- parse grobid text
        print("parse by grobid ..")
        if path_exits(join(output_dir, paper_id)):
            continue
        mkdir(join(output_dir, paper_id))
        pdf_parser_command = "python -m scirex_utilities.convert_pdf_to_prediction_input " + input_dir + " " \
                             + join(output_dir, str(paper_id)) + " " + str(paper_id)

        parser_result = subprocess.run(pdf_parser_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True)

        if parser_result.returncode != 0:
            print("Parsing failed ...")
            print(parser_result.stderr)
            continue

        print("Resolve to relations")
        print("--------------------------")
        relations_cmd = "test_file=" + join(join(output_dir, str(paper_id)),
                                            str(paper_id) + "_scirex_prediction_input.json") \
                        + " test_output_folder=" + join(output_dir, str(paper_id)) \
                        + " paper_id=" + str(paper_id) \
                        + " scirex_archive=outputs/pwc_outputs/experiment_scirex_full/main" \
                        + " scirex_coreference_archive=outputs/pwc_outputs/experiment_coreference/main" \
                        + " cuda_device=" + str(cuda_device) \
                        + " bash scirex/commands/predict_external_pdf_paper.sh"

        relation_result = subprocess.run(relations_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         text=True)

        if relation_result.returncode != 0:
            print("Relation failed ...")
            print(relation_result.stderr)
            continue


if __name__ == '__main__':
    parser = ArgumentParser("Convert a list of pdf files to resolved relation.")
    parser.add_argument("input_dir", help="The input directory of the pdfs paper .")
    parser.add_argument("output_dir", help="The output directory of the result, a directory will be created per pdf.")
    parser.add_argument("cuda_device", help="The cuda device id")

    args = parser.parse_args()
    main(args)
