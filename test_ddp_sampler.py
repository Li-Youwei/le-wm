import unittest

import torch


class DistributedWeightedSamplerTest(unittest.TestCase):
    def test_ranks_slice_one_shared_weighted_draw(self) -> None:
        from ddp_sampler import DistributedWeightedSampler

        weights = torch.arange(1, 10, dtype=torch.float64)
        samplers = [
            DistributedWeightedSampler(
                weights,
                num_samples=20,
                replacement=True,
                num_replicas=2,
                rank=rank,
                seed=1234,
            )
            for rank in (0, 1)
        ]

        rank0 = list(iter(samplers[0]))
        rank1 = list(iter(samplers[1]))

        self.assertEqual(len(rank0), 10)
        self.assertEqual(len(rank1), 10)
        self.assertEqual(len(rank0) + len(rank1), 20)
        self.assertNotEqual(rank0, rank1)

    def test_same_seed_and_epoch_are_reproducible(self) -> None:
        from ddp_sampler import DistributedWeightedSampler

        weights = torch.arange(1, 10, dtype=torch.float64)
        first = DistributedWeightedSampler(
            weights,
            num_samples=20,
            replacement=True,
            num_replicas=2,
            rank=0,
            seed=99,
        )
        second = DistributedWeightedSampler(
            weights,
            num_samples=20,
            replacement=True,
            num_replicas=2,
            rank=0,
            seed=99,
        )

        self.assertEqual(list(iter(first)), list(iter(second)))

    def test_different_epoch_changes_sampling(self) -> None:
        from ddp_sampler import DistributedWeightedSampler

        weights = torch.arange(1, 10, dtype=torch.float64)
        sampler = DistributedWeightedSampler(
            weights,
            num_samples=20,
            replacement=True,
            num_replicas=2,
            rank=0,
            seed=99,
        )
        epoch0 = list(iter(sampler))
        sampler.set_epoch(1)
        epoch1 = list(iter(sampler))

        self.assertNotEqual(epoch0, epoch1)

    def test_different_iteration_changes_sampling(self) -> None:
        from ddp_sampler import DistributedWeightedSampler

        weights = torch.arange(1, 10, dtype=torch.float64)
        sampler = DistributedWeightedSampler(
            weights,
            num_samples=20,
            replacement=True,
            num_replicas=2,
            rank=0,
            seed=99,
        )

        self.assertNotEqual(list(iter(sampler)), list(iter(sampler)))

    def test_index_sampler_pads_tiny_dataset_for_all_ranks(self) -> None:
        from ddp_sampler import DistributedIndexSampler

        ranks = [
            list(
                DistributedIndexSampler(
                    dataset_size=1,
                    shuffle=False,
                    num_replicas=4,
                    rank=rank,
                )
            )
            for rank in range(4)
        ]

        self.assertEqual(ranks, [[0], [0], [0], [0]])


if __name__ == "__main__":
    unittest.main()
