SUBOBJECTS_MAP = {
        'person': [
            {'name': 'clock', 'min_iou': 0.01},     
            {'name': 'cell phone', 'min_iou': 0.01},  
            {'name': 'backpack', 'min_iou': 0.05},
            {'name': 'glasses', 'min_iou': 0.01},
            {'name': 'toothbrush', 'min_iou': 0.01},
            {'name': 'pencil', 'min_iou': 0.01},
            {'name': 'pen', 'min_iou': 0.01}
        ],
        'dining table': [
            {'name': 'laptop', 'min_iou': 0.01},     
            {'name': 'cell phone', 'min_iou': 0.05},  
            {'name': 'bottle', 'min_iou': 0.01},  
            {'name': 'backpack', 'min_iou': 0.05},
        ],
        'car': [
            {'name': 'license plate', 'min_iou': 0.01},
            {'name': 'wheel', 'min_iou': 0.05}
        ],
        'bicycle': [
            {'name': 'wheel', 'min_iou': 0.05},
            {'name': 'person', 'min_iou': 0.1}
        ],
        'motorcycle': [
            {'name': 'wheel', 'min_iou': 0.05},
            {'name': 'person', 'min_iou': 0.1}
        ]
    }
