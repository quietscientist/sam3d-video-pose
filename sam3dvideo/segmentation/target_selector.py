#!/usr/bin/env python3
"""Continuity-based single-target selection for SAM3 video outputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


@dataclass
class TargetLockDecision:
    """Small debug record for target-lock decisions."""

    selected: bool
    frame_idx: int
    reason: str
    source_object_id: int | None = None
    score: float | None = None
    iou: float | None = None
    center_dist_norm: float | None = None
    should_log: bool = False


@dataclass
class _Candidate:
    index: int
    object_id: int
    score: float
    mask: torch.Tensor
    area: float
    center: tuple[float, float]
    box: tuple[float, float, float, float]
    iou: float = 0.0
    center_dist_norm: float = math.inf
    area_ratio: float = math.inf
    target_score: float = -math.inf
    accepted: bool = False


class TargetSelector:
    """Select one stable target from one or more SAM3 masklets."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        output_object_id: int = 0,
        initial_selection: str = "score_center",
        min_iou: float = 0.02,
        max_center_jump: float = 0.15,
        max_area_ratio: float = 4.0,
        reacquire_after: int = 12,
        center_prior_weight: float = 0.25,
    ):
        self.enabled = enabled
        self.output_object_id = output_object_id
        self.initial_selection = initial_selection
        self.min_iou = min_iou
        self.max_center_jump = max_center_jump
        self.max_area_ratio = max_area_ratio
        self.reacquire_after = reacquire_after
        self.center_prior_weight = center_prior_weight

        self.last_mask: torch.Tensor | None = None
        self.last_center: tuple[float, float] | None = None
        self.last_velocity: tuple[float, float] = (0.0, 0.0)
        self.last_area: float | None = None
        self.last_frame_idx: int | None = None
        self.last_source_object_id: int | None = None
        self.missed_frames = 0

    def select(self, outputs: dict[str, Any], frame_idx: int) -> tuple[dict[str, Any], TargetLockDecision]:
        """Filter SAM3 outputs down to a single continuity-checked target."""
        if not self.enabled:
            return outputs, TargetLockDecision(True, frame_idx, "disabled")

        candidates = self._build_candidates(outputs)
        if not candidates:
            self.missed_frames += 1
            decision = TargetLockDecision(
                False,
                frame_idx,
                "no_candidates",
                should_log=self._should_log_miss(),
            )
            return self._empty_like(outputs), decision

        if self.last_mask is None or self.last_center is None or self.last_area is None:
            selected = self._select_initial(candidates)
            self._update_state(selected, frame_idx)
            return self._filter_outputs(outputs, selected), TargetLockDecision(
                True,
                frame_idx,
                "initial_lock",
                source_object_id=selected.object_id,
                score=selected.score,
                should_log=True,
            )

        selected = self._select_by_continuity(candidates, frame_idx)
        if selected is None:
            self.missed_frames += 1
            best = max(candidates, key=lambda c: c.target_score)
            decision = TargetLockDecision(
                False,
                frame_idx,
                "rejected_jump",
                source_object_id=best.object_id,
                score=best.score,
                iou=best.iou,
                center_dist_norm=best.center_dist_norm,
                should_log=self._should_log_miss(),
            )
            return self._empty_like(outputs), decision

        previous_source = self.last_source_object_id
        had_misses = self.missed_frames > 0
        self._update_state(selected, frame_idx)
        reason = "source_changed" if previous_source is not None and selected.object_id != previous_source else "locked"
        return self._filter_outputs(outputs, selected), TargetLockDecision(
            True,
            frame_idx,
            reason,
            source_object_id=selected.object_id,
            score=selected.score,
            iou=selected.iou,
            center_dist_norm=selected.center_dist_norm,
            should_log=reason == "source_changed" or had_misses,
        )

    def _build_candidates(self, outputs: dict[str, Any]) -> list[_Candidate]:
        masks = outputs.get("masks")
        object_ids = outputs.get("object_ids")
        scores = outputs.get("scores")
        boxes = outputs.get("boxes")
        if masks is None or object_ids is None or scores is None or len(masks) == 0:
            return []

        candidates = []
        for idx in range(len(masks)):
            mask = masks[idx].detach().cpu().bool()
            area = float(mask.sum().item())
            if area <= 0:
                continue

            box_tensor = boxes[idx].detach().cpu().float() if boxes is not None and len(boxes) > idx else None
            box = self._box_tuple(box_tensor, mask)
            center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
            candidates.append(
                _Candidate(
                    index=idx,
                    object_id=int(object_ids[idx].item()),
                    score=float(scores[idx].item()),
                    mask=mask,
                    area=area,
                    center=center,
                    box=box,
                )
            )
        return candidates

    def _select_initial(self, candidates: list[_Candidate]) -> _Candidate:
        if self.initial_selection == "largest":
            return max(candidates, key=lambda c: c.area)
        if self.initial_selection == "center":
            return max(candidates, key=self._center_prior_score)
        return max(
            candidates,
            key=lambda c: c.score + self.center_prior_weight * self._center_prior_score(c),
        )

    def _select_by_continuity(self, candidates: list[_Candidate], frame_idx: int) -> _Candidate | None:
        expected_center = self._expected_center(frame_idx)
        last_area = max(self.last_area or 1.0, 1.0)
        diag = self._mask_diag(candidates[0].mask)
        allow_reacquire = self.missed_frames >= self.reacquire_after

        for candidate in candidates:
            candidate.iou = self._mask_iou(candidate.mask, self.last_mask)
            candidate.center_dist_norm = math.dist(candidate.center, expected_center) / diag
            candidate.area_ratio = max(candidate.area / last_area, last_area / max(candidate.area, 1.0))
            center_score = max(0.0, 1.0 - candidate.center_dist_norm / max(self.max_center_jump, 1e-6))
            area_limit = max(self.max_area_ratio, 1.01)
            area_penalty = min(1.0, math.log(max(candidate.area_ratio, 1.0)) / math.log(area_limit))
            candidate.target_score = (
                4.0 * candidate.iou
                + 2.0 * center_score
                + 0.5 * candidate.score
                - 0.75 * area_penalty
            )
            candidate.accepted = (
                candidate.iou >= self.min_iou
                or (
                    candidate.center_dist_norm <= self.max_center_jump
                    and candidate.area_ratio <= self.max_area_ratio
                )
                or allow_reacquire
            )

        accepted = [candidate for candidate in candidates if candidate.accepted]
        if not accepted:
            return None
        return max(accepted, key=lambda c: c.target_score)

    def _update_state(self, candidate: _Candidate, frame_idx: int):
        if self.last_center is not None and self.last_frame_idx is not None:
            frame_delta = max(frame_idx - self.last_frame_idx, 1)
            self.last_velocity = (
                (candidate.center[0] - self.last_center[0]) / frame_delta,
                (candidate.center[1] - self.last_center[1]) / frame_delta,
            )
        self.last_mask = candidate.mask
        self.last_center = candidate.center
        self.last_area = candidate.area
        self.last_frame_idx = frame_idx
        self.last_source_object_id = candidate.object_id
        self.missed_frames = 0

    def _expected_center(self, frame_idx: int) -> tuple[float, float]:
        if self.last_center is None or self.last_frame_idx is None:
            return (0.0, 0.0)
        frame_delta = max(frame_idx - self.last_frame_idx, 1)
        return (
            self.last_center[0] + self.last_velocity[0] * frame_delta,
            self.last_center[1] + self.last_velocity[1] * frame_delta,
        )

    def _filter_outputs(self, outputs: dict[str, Any], candidate: _Candidate) -> dict[str, Any]:
        filtered = dict(outputs)
        idx = candidate.index
        for key in ("masks", "scores", "boxes"):
            value = outputs.get(key)
            if torch.is_tensor(value):
                filtered[key] = value[idx : idx + 1]

        object_ids = outputs.get("object_ids")
        if torch.is_tensor(object_ids):
            filtered["object_ids"] = torch.tensor(
                [self.output_object_id],
                dtype=object_ids.dtype,
                device=object_ids.device,
            )

        prompt_to_obj_ids = outputs.get("prompt_to_obj_ids", {})
        filtered["prompt_to_obj_ids"] = {
            prompt: [self.output_object_id]
            for prompt, obj_ids in prompt_to_obj_ids.items()
            if candidate.object_id in obj_ids
        }
        return filtered

    def _empty_like(self, outputs: dict[str, Any]) -> dict[str, Any]:
        filtered = dict(outputs)
        for key in ("masks", "object_ids", "scores", "boxes"):
            value = outputs.get(key)
            if torch.is_tensor(value):
                filtered[key] = value[:0]
        filtered["prompt_to_obj_ids"] = {}
        return filtered

    def _center_prior_score(self, candidate: _Candidate) -> float:
        height, width = candidate.mask.shape[-2:]
        diag = math.hypot(width, height)
        image_center = (width / 2.0, height / 2.0)
        return max(0.0, 1.0 - math.dist(candidate.center, image_center) / (0.5 * diag))

    def _should_log_miss(self) -> bool:
        return self.missed_frames == 1 or self.missed_frames % 30 == 0

    @staticmethod
    def _box_tuple(box_tensor: torch.Tensor | None, mask: torch.Tensor) -> tuple[float, float, float, float]:
        if box_tensor is not None and box_tensor.numel() == 4:
            x1, y1, x2, y2 = [float(v) for v in box_tensor.tolist()]
            if x2 > x1 and y2 > y1:
                return x1, y1, x2, y2

        ys, xs = torch.nonzero(mask, as_tuple=True)
        if len(xs) == 0:
            return 0.0, 0.0, 0.0, 0.0
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    @staticmethod
    def _mask_iou(mask: torch.Tensor, last_mask: torch.Tensor | None) -> float:
        if last_mask is None or mask.shape != last_mask.shape:
            return 0.0
        intersection = torch.logical_and(mask, last_mask).sum().float()
        union = torch.logical_or(mask, last_mask).sum().float().clamp(min=1.0)
        return float((intersection / union).item())

    @staticmethod
    def _mask_diag(mask: torch.Tensor) -> float:
        height, width = mask.shape[-2:]
        return max(math.hypot(width, height), 1.0)
