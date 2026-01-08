def check_precision_in_text(pred_text, keywords):
    trigger_map = {
        "Baby visible": ["newborn","infant"],
        "Ventilation": ["CPAP","continuous positive airway pressure", "positive pressure ventilation", "PPV", "ventilation"],
        "Suction": ["suctioning", "suction"],
        "Stimulation": ["stimulation"],
    }
    pred_text_lower = pred_text.lower()
    precision_dict = {}

    predicted_classes = set()
    for class_name, triggers in trigger_map.items():
        for kw in triggers:
            if kw.lower() in pred_text_lower:
                predicted_classes.add(class_name)
                break
    
    for pred_class in predicted_classes:
        is_correct = False
        for gt in keywords:
            if gt.startswith("P-"):
                gt = gt[2:]
            if pred_class == gt:
                is_correct = True
                break
        precision_dict[pred_class] = 1 if is_correct else 0
    return precision_dict

def check_recall_in_text(pred_text, keywords):
    keyword_map = {
        "Baby visible": ["newborn","infant"],
        "Ventilation": ["CPAP","continuous positive airway pressure", "positive pressure ventilation", "PPV", "ventilation"],
        "Suction": ["suctioning", "suction"],
        "Stimulation": ["stimulation"],
    }
    pred_text_lower = pred_text.lower()
    recall_dict = {}
    for gt in keywords:
        if gt.startswith("P-"):
            gt = gt[2:]
        class_detected = False
        for kw in keyword_map[gt]:
            if kw.lower() in pred_text_lower:
                class_detected = True
                break
        recall_dict[gt] = 1 if class_detected else 0
    return recall_dict