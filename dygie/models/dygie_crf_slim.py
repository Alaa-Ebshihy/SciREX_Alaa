import copy
import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.span_extractors import EndpointSpanExtractor, SelfAttentiveSpanExtractor
from dygie.models.span_extractor import MaxPoolSpanExtractor
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util

# Import submodules.
from dygie.models.coref import CorefResolver
from dygie.models.ner_crf_tagger_slim import NERTagger
from dygie.models.relation_pwc_crf import RelationExtractor
from dygie.models.span_classifier_binary import SpanClassifier
from dygie.models.span_classifier import SpanClassifier as MultiClassSpanClassifier

from dygie.training.ner_metrics import NERMentionMetrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("dygie_crf_slim")
class DyGIECRFSlim(Model):
    """
    TODO(dwadden) document me.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``
        This layer incorporates contextual information for each word in the document.
    feature_size: ``int``
        The embedding size for all the embedded features, such as distances or span widths.
    submodule_params: ``TODO(dwadden)``
        A nested dictionary specifying parameters to be passed on to initialize submodules.
    max_span_width: ``int``
        The maximum width of candidate spans.
    lexical_dropout: ``int``
        The probability of dropping out dimensions of the embedded text.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the reglarization penalty during training.
    display_metrics: ``List[str]``. A list of the metrcs that should be printed out during model
        training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        residual_text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        modules,
        feature_size: int,
        loss_weights: Dict[str, int],
        max_span_width: int = 20,
        lexical_dropout: float = 0.2,
        use_attentive_span_extractor: bool = True,
        use_entity_classifier: bool = True,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        display_metrics: List[str] = None,
    ) -> None:
        super(DyGIECRF, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._residual_text_field_embedder = residual_text_field_embedder
        self._context_layer = context_layer

        self._loss_weights = loss_weights.as_dict()
        self._permanent_loss_weights = copy.deepcopy(self._loss_weights)

        self._coref = CorefResolver.from_params(vocab=vocab, feature_size=feature_size, params=modules.pop("coref"))
        self._ner = NERTagger.from_params(vocab=vocab, params=modules.pop("ner"))
        self._relation = RelationExtractor.from_params(
            vocab=vocab, feature_size=feature_size, params=modules.pop("relation")
        )

        self._link_classifier = SpanClassifier.from_params(vocab=vocab, params=modules.pop("link_classifier"))

        self._use_entity_classifier = use_entity_classifier
        if use_entity_classifier:
            self._entity_classifier = MultiClassSpanClassifier.from_params(
                vocab=vocab, params=modules.pop("entity_classifier")
            )

        self._endpoint_span_extractor = EndpointSpanExtractor(
            context_layer.get_output_dim(),
            combination="x,y",
            num_width_embeddings=max_span_width,
            span_width_embedding_dim=feature_size,
            bucket_widths=False,
        )

        if use_attentive_span_extractor:
            self._attentive_span_extractor = SelfAttentiveSpanExtractor(input_dim=context_layer.get_output_dim())
        else:
            self._attentive_span_extractor = None

        self._max_span_width = max_span_width

        self._display_metrics = display_metrics

        self._ner_entity_metrics = NERMentionMetrics(['Material', 'Method', 'Task', 'Metric'])
        self._ner_mention_metrics = NERMentionMetrics(['Entity'])

        if lexical_dropout > 0:
            self._lexical_dropout = torch.nn.Dropout(p=lexical_dropout)
        else:
            self._lexical_dropout = lambda x: x

        initializer(self)

    @overrides
    def forward(
        self,
        text,
        ner_labels,
        ner_entity_labels,
        ner_link_labels,
        ner_is_entity_labels,
        spans,
        span_coref_labels,
        span_link_labels,
        span_entity_labels,
        relation_index,
        metadata,
    ):

        # Shape: (batch_size, max_sentence_length, embedding_size)
        text_embeddings = self._lexical_dropout(self._text_field_embedder(text))
        residual_text_embeddings = self._lexical_dropout(self._residual_text_field_embedder(text))

        text_embeddings = torch.cat([text_embeddings, residual_text_embeddings], dim=-1)
        text_embeddings, text_mask, sentence_lengths = self.extract_sentence_from_context(metadata, text_embeddings)

        # Shape: (batch_size, max_sentence_length, encoding_dim)
        flat_text_embeddings = text_embeddings.view(-1, text_embeddings.size(-1))
        flat_text_mask = text_mask.view(-1).byte()

        filtered_text_embeddings = flat_text_embeddings[flat_text_mask.byte()]
        filtered_contextualized_embeddings = self._context_layer(
            filtered_text_embeddings.unsqueeze(0),
            torch.ones((1, filtered_text_embeddings.size(0)), device=filtered_text_embeddings.device).byte(),
        ).squeeze(0)

        flat_contextualized_embeddings = torch.zeros(
            (flat_text_embeddings.size(0), filtered_contextualized_embeddings.size(1)),
            device=filtered_text_embeddings.device,
        )
        flat_contextualized_embeddings.masked_scatter_(flat_text_mask.unsqueeze(-1), filtered_contextualized_embeddings)
        contextualized_embeddings = flat_contextualized_embeddings.reshape(
            (text_embeddings.size(0), text_embeddings.size(1), flat_contextualized_embeddings.size(-1))
        )

        # Make calls out to the modules to get results.
        output_ner = {"loss": 0}
        output_relation = {"loss": 0, "metadata": metadata}
        output_linker = {"loss": 0}
        output_entity = {"loss": 0}

        ner_labels_dispatcher = {
            "ner_labels": ner_labels,
            "ner_entity_labels": ner_entity_labels,
            "ner_link_labels": ner_link_labels,
            "ner_is_entity_labels": ner_is_entity_labels,
        }

        output_ner = self._ner(
            contextualized_embeddings, text_mask, ner_labels_dispatcher[self._ner.label_namespace], metadata
        )

        output_ner = self._ner.decode(output_ner)
        assert "spans" in output_ner

        predicted_spans = output_ner["spans"].to(text_embeddings.device).long()

        if self._use_entity_classifier:
            if predicted_spans.nelement() != 0 :
                predicted_spans_ec, predicted_span_mask, predicted_span_embeddings = self.generate_span_embeddings(
                    contextualized_embeddings, predicted_spans
                )

                output_predicted_entity = self._entity_classifier(
                    predicted_spans_ec, predicted_span_mask, predicted_span_embeddings, metadata=metadata
                )

                output_predicted_entity = self._entity_classifier.decode(output_predicted_entity)

                decoded_entity_spans = output_predicted_entity['decoded_spans']
                decoded_ner_spans = output_ner['decoded_ner']

                assert len(decoded_entity_spans) == len(decoded_ner_spans)
                for entity_dict, ner_dict in zip(decoded_entity_spans, decoded_ner_spans):
                    assert len(entity_dict) == len(ner_dict)
                    for k, v in ner_dict.items() :
                        assert k in entity_dict
                        v.append(entity_dict[k])

            gold_entity_dict = [{k: v[1] for k, v in x['ner_dict'].items()} for x in metadata]
            gold_ner_dict = output_ner['gold_ner']
            assert len(gold_entity_dict) == len(gold_ner_dict)
            for entity_dict, ner_dict in zip(gold_entity_dict, gold_ner_dict):
                assert len(entity_dict) == len(ner_dict)
                for k, v in ner_dict.items() :
                    assert k in entity_dict
                    v.append(entity_dict[k])

        self.add_mention_metrics(output_ner['decoded_ner'], output_ner['gold_ner'])

        if spans is None:
            spans = predicted_spans
            if self._use_entity_classifier and spans.nelement() != 0:
                span_entity_labels = output_predicted_entity['ner_probs'].argmax(-1)

        if spans.nelement() != 0:
            spans, span_mask, span_embeddings = self.generate_span_embeddings(contextualized_embeddings, spans)

            if span_mask.sum() != 0 :
                # Entity_classifier
                if self._use_entity_classifier and span_link_labels is not None:
                    output_entity = self._entity_classifier(
                        spans, span_mask, span_embeddings, span_entity_labels, metadata=metadata
                    )

                # Add Position Info to Span Embeddings
                start_pos_in_doc = torch.LongTensor([x["start_pos_in_doc"] for x in metadata]).to(spans.device)
                sentence_offset = start_pos_in_doc.unsqueeze(1).unsqueeze(2)
                span_offset = spans + (sentence_offset * span_mask.unsqueeze(-1).long())

                doc_length = metadata[0]["document_metadata"]["doc_length"]
                span_position = span_offset.float().mean(-1, keepdim=True) / doc_length

                span_embeddings = torch.cat([span_embeddings, span_position], dim=-1)

                # Add Entity Labels to Span Embeddings
                n_entity_labels = self.vocab.get_vocab_size("span_entity_labels")
                span_entity_labels_one_hot = torch.zeros(
                    (span_entity_labels.size(0), span_entity_labels.size(1), n_entity_labels)
                ).to(spans.device)
                span_entity_labels_one_hot.scatter_(-1, span_entity_labels.unsqueeze(-1), 1)

                link_embeddings = torch.cat([span_embeddings, span_entity_labels_one_hot], dim=-1)

                # Linking
                output_linker = self._link_classifier(
                    spans, span_mask, link_embeddings, span_link_labels, metadata=metadata
                )

                # Relation Extraction
                output_relation = self._relation.compute_representations(
                    spans, span_mask, link_embeddings, span_coref_labels, relation_index, metadata
                )
                output_relation = self._relation.predict_labels(output_relation)

        loss = output_ner["loss"] + output_relation["loss"] + output_linker["loss"] + output_entity["loss"]

        output_dict = dict(relation=output_relation, ner=output_ner, linked=output_linker, entity=output_entity)
        output_dict["loss"] = loss

        return output_dict

    def generate_span_embeddings(self, contextualized_embeddings, spans):
        attended_span_embeddings = self._attentive_span_extractor(contextualized_embeddings, spans)
        span_mask = (spans[:, :, 0] >= 0).float()
        spans = F.relu(spans.float()).long()
        endpoint_span_embeddings = self._endpoint_span_extractor(contextualized_embeddings, spans)
        span_embeddings = torch.cat([endpoint_span_embeddings, attended_span_embeddings], -1)
        return spans, span_mask, span_embeddings

    def extract_sentence_from_context(self, metadata, text_embeddings):
        sentence_spans = torch.LongTensor([[x["start_ix"], x["end_ix"]] for x in metadata]).to(text_embeddings.device)
        sentence_lengths = sentence_spans[:, 1] - sentence_spans[:, 0]
        max_sentence_length = sentence_lengths.max()

        range_vector = util.get_range_vector(max_sentence_length, util.get_device_of(text_embeddings)).view(1, -1)
        span_indices = torch.clamp_max(sentence_spans[:, 0:1] + range_vector, text_embeddings.shape[1] - 1)
        text_mask = util.get_mask_from_sequence_lengths(sentence_lengths, max_length=max_sentence_length)
        text_embeddings = util.batched_index_select(text_embeddings, span_indices) * text_mask.unsqueeze(-1).float()
        return text_embeddings, text_mask, sentence_lengths

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]):
        res = {}
        res["ner"] = output_dict['ner']
        res["relation"] = self._relation.decode(output_dict["relation"])
        res["linked"] = self._link_classifier.decode(output_dict["linked"])

        return res

    def add_mention_metrics(self, prediction_dict, gold_dict) :
        get_n = lambda x, n : [{k:v[n] for k, v in d.items()} for d in x]
        
        self._ner_mention_metrics(get_n(prediction_dict, 0), get_n(gold_dict, 0))
        self._ner_entity_metrics(get_n(prediction_dict, 1), get_n(gold_dict, 1))

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        Get all metrics from all modules. For the ones that shouldn't be displayed, prefix their
        keys with an underscore.
        """
        metrics_ner_entity = self._ner_entity_metrics.get_metric(reset=reset)
        metrics_ner_mention = self._ner_mention_metrics.get_metric(reset=reset)

        metrics_ner_entity = {'ner_' + k:v for k, v in metrics_ner_entity.items()}
        metrics_ner_mention = {'ner_mention_' + k:v for k, v in metrics_ner_mention.items()}

        metrics_ner = dict(list(metrics_ner_entity.items()) + list(metrics_ner_mention.items()))
        metrics_relation = self._relation.get_metrics(reset=reset)
        metrics_link = self._link_classifier.get_metrics(reset=reset)

        if self._use_entity_classifier :
            metrics_ner.update(self._entity_classifier.get_metrics(reset=reset))

        # Make sure that there aren't any conflicting names.
        metric_names = list(metrics_ner.keys()) + list(metrics_relation.keys()) + list(metrics_link.keys())
        assert len(set(metric_names)) == len(metric_names)
        all_metrics = dict(list(metrics_ner.items()) + list(metrics_relation.items()) + list(metrics_link.items()))

        # If no list of desired metrics given, display them all.
        if self._display_metrics is None:
            return all_metrics
        # Otherwise only display the selected ones.
        res = {}
        for k, v in all_metrics.items():
            if k in self._display_metrics:
                res[k] = v
            else:
                new_k = "_" + k
                res[new_k] = v
        return res
