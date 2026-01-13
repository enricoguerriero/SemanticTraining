def get_broad_class_map():
    """
    Maps specific Ground Truth labels to the broad classes 
    used in the text detection logic.
    """
    return {
        "Baby visible":           "Baby visible",
        "CPAP":                   "Ventilation",
        "PPV":                    "Ventilation",
        "Suction":                "Suction",
        "Stimulation extremities":"Stimulation",
        "Stimulation backnates":  "Stimulation",
        "Stimulation trunk":      "Stimulation",
    }

def check_precision_in_text(pred_text, keywords):
    trigger_map = {
        "Baby visible": ["newborn","infant"],
        "Ventilation": ["CPAP","continuous positive airway pressure", "positive pressure ventilation", "PPV", "ventilation"],
        "Suction": ["suctioning", "suction"],
        "Stimulation": ["stimulation"],
    }
    gt_map = get_broad_class_map()
    broad_ground_truths = set()

    for k in keywords:
        clean_k = k[2:] if k.startswith("P-") else k
        if clean_k in gt_map:
            broad_ground_truths.add(gt_map[clean_k])

    pred_text_lower = pred_text.lower()
    precision_dict = {}

    predicted_classes = set()
    for class_name, triggers in trigger_map.items():
        for kw in triggers:
            if kw.lower() in pred_text_lower:
                predicted_classes.add(class_name)
                break
    
    for pred_class in predicted_classes:
        is_correct = pred_class in broad_ground_truths
        precision_dict[pred_class] = 1 if is_correct else 0
        
    return precision_dict

def check_recall_in_text(pred_text, keywords):
    trigger_map = {
        "Baby visible": ["newborn","infant"],
        "Ventilation": ["CPAP","continuous positive airway pressure", "positive pressure ventilation", "PPV", "ventilation"],
        "Suction": ["suctioning", "suction"],
        "Stimulation": ["stimulation"],
    }
    gt_map = get_broad_class_map()
    pred_text_lower = pred_text.lower()
    recall_dict = {}

    for gt in keywords:
        clean_gt = gt[2:] if gt.startswith("P-") else gt
        
        if clean_gt not in gt_map:
            continue
            
        broad_category = gt_map[clean_gt]
        
        class_detected = False
        if broad_category in trigger_map:
            for kw in trigger_map[broad_category]:
                if kw.lower() in pred_text_lower:
                    class_detected = True
                    break
        
        recall_dict[broad_category] = 1 if class_detected else 0
        
    return recall_dict