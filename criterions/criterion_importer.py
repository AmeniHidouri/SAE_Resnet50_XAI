from criterions.classification_criterion import ClassificationCriterion
from criterions.complexity_criterion import ComplexityCriterion
from criterions.faithfullness_criterion import FaithfullnessCriterion
from criterions.max_sensivity_criterion import MaxSensitivityCriterion
from criterions.mllm_criterion import MLLMCriterion
from criterions.sim_bbox import SimBboxes
from criterions.sim_cbm import SimCBM
from criterions.variance_criterion import VarianceCriterion


def criterion_importer(
    criterion_name,
    model,
    metadata_dataset,
    expl_method,
    device,
    type_expl="saliency",
    xai_id=-1,
    dataset_name="",
):
    if criterion_name == "variance_gauss":
        return VarianceCriterion(
            model, 3, expl_method, device, xai_id=xai_id, dataset_name=dataset_name
        )
    if criterion_name == "variance_sharp":
        return VarianceCriterion(
            model, 8, expl_method, device, xai_id=xai_id, dataset_name=dataset_name
        )
    if criterion_name == "variance_bright":
        return VarianceCriterion(
            model, 5, expl_method, device, xai_id=xai_id, dataset_name=dataset_name
        )
    if criterion_name == "complexity_10":
        return ComplexityCriterion(10, metadata_dataset)
    if criterion_name == "complexity_20":
        return ComplexityCriterion(20, metadata_dataset)
    if criterion_name == "complexity_30":
        return ComplexityCriterion(30, metadata_dataset)
    if criterion_name == "classification_logistic":
        if type_expl == "saliency":
            return ClassificationCriterion(metadata_dataset, device, "CLIP-few-shot-logistic")
        if type_expl == "cbm":
            return ClassificationCriterion(metadata_dataset, device, "logistic")
    if criterion_name == "classification_qda":
        if type_expl == "saliency":
            return ClassificationCriterion(metadata_dataset, device, "CLIP-few-shot-qda")
        if type_expl == "cbm":
            return ClassificationCriterion(metadata_dataset, device, "qda")
    if criterion_name == "classification_svm":
        if type_expl == "saliency":
            return ClassificationCriterion(metadata_dataset, device, "CLIP-few-shot-svm")
        if type_expl == "cbm":
            return ClassificationCriterion(metadata_dataset, device, "svm")
    if criterion_name == "faithfullness":
        return FaithfullnessCriterion(model, device, metadata_dataset)
    if criterion_name == "maxsensitivity":
        return MaxSensitivityCriterion(model, device, metadata_dataset, expl_method)
    if criterion_name == "mllm":
        return MLLMCriterion(model, expl_method)
    if criterion_name == "saliencysum":
        return SimBboxes("all")
    if criterion_name == "saliencytruth":
        return SimBboxes("in")
    if criterion_name == "saliencytruthback":
        return SimBboxes("out")
    if criterion_name == "saliencyentropy":
        return SimBboxes("entropy")
    if criterion_name == "cbmbleu":
        return SimCBM("bleu")
    if criterion_name == "cbmrouge":
        return SimCBM("rouge")
    if criterion_name == "cbmentropy":
        return SimCBM("entropy")
    raise ValueError("Criterion not implemented !")
