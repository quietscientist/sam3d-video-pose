#!/usr/bin/env python3
"""
SAM3 model patching utilities.
"""

import transformers.models.sam3.modeling_sam3 as sam3_module

PATCHED = False


def patch_sam3():
    """Patch Sam3Model.forward to handle CLIPTextModelOutput"""
    global PATCHED

    if PATCHED:
        return

    original_sam3_forward = sam3_module.Sam3Model.forward

    def patched_sam3_forward(self, pixel_values=None, vision_embeds=None, input_ids=None,
                            attention_mask=None, text_embeds=None, input_boxes=None,
                            input_boxes_labels=None, **kwargs):
        if text_embeds is not None:
            if hasattr(text_embeds, 'pooler_output'):
                text_embeds = text_embeds.pooler_output
            elif hasattr(text_embeds, 'last_hidden_state'):
                text_embeds = text_embeds.last_hidden_state

        return original_sam3_forward(
            self, pixel_values, vision_embeds, input_ids, attention_mask,
            text_embeds, input_boxes, input_boxes_labels, **kwargs
        )

    sam3_module.Sam3Model.forward = patched_sam3_forward
    PATCHED = True
    print("✓ SAM3 patched successfully")
