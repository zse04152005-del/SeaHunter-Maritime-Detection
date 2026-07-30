from __future__ import annotations

import unittest

from seahunter.innovation import (
    ResourceState,
    ROIProposal,
    ROISource,
    ScaleDetection,
    SelectiveROIConfig,
    SelectiveROIScheduler,
    fuse_scale_detections,
)


class SelectiveROIInnovationTests(unittest.TestCase):
    def test_proposals_merge_rank_and_refresh_periodically(self) -> None:
        scheduler = SelectiveROIScheduler(SelectiveROIConfig(top_k=2, global_refresh_interval=10))
        proposals = (
            ROIProposal((10.0, 10.0, 30.0, 30.0), 0.9, ROISource.OBJECTNESS),
            ROIProposal((12.0, 12.0, 31.0, 31.0), 0.8, ROISource.MOTION),
            ROIProposal((100.0, 100.0, 130.0, 130.0), 0.7, ROISource.LOST_TRACK),
        )
        plan = scheduler.plan(
            frame_index=20,
            frame_size=(1920, 1080),
            proposals=proposals,
            resources=ResourceState(20.0, 30.0, 60.0, 0),
        )

        self.assertTrue(plan.global_refresh)
        self.assertEqual(len(plan.rois), 2)
        self.assertEqual(plan.rois[0].source, ROISource.OBJECTNESS)
        self.assertEqual(plan.roi_resolution, 1024)

    def test_latency_temperature_and_queue_reduce_bounded_budget(self) -> None:
        scheduler = SelectiveROIScheduler(SelectiveROIConfig(top_k=5, minimum_top_k=2))
        proposals = tuple(
            ROIProposal((float(index * 20), 0.0, float(index * 20 + 10), 10.0), 1.0 - index * 0.1, ROISource.MOTION)
            for index in range(5)
        )
        plan = scheduler.plan(
            frame_index=1,
            frame_size=(200, 100),
            proposals=proposals,
            resources=ResourceState(60.0, 30.0, 85.0, 8),
        )

        self.assertEqual(plan.roi_budget, 2)
        self.assertEqual(plan.roi_resolution, 768)
        self.assertEqual(set(plan.degradation_reasons), {"latency_pressure", "queue_pressure", "thermal_pressure"})

    def test_cross_scale_fusion_removes_boundary_duplicates(self) -> None:
        detections = (
            ScaleDetection((0.0, 0.0, 10.0, 10.0), 1, 0.8, "global"),
            ScaleDetection((1.0, 1.0, 11.0, 11.0), 1, 0.9, "roi"),
            ScaleDetection((1.0, 1.0, 11.0, 11.0), 2, 0.7, "roi"),
        )
        fused = fuse_scale_detections(detections, iou_threshold=0.6)

        self.assertEqual(len(fused), 2)
        self.assertEqual(fused[0].source, "roi")


if __name__ == "__main__":
    unittest.main()
