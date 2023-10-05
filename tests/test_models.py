import torch
import numpy as np
import unittest


from modules import *


N_CHANNELS = [160, 204, 298]
N_SAMPLES = [16, 32, 64, 128]

REPLICATION_INPUT_SIZES = [(c, t) for c in N_CHANNELS for t in N_SAMPLES]


class TestAutoencoder(unittest.TestCase):
    def test_Encoder(self):
        for channels, samples in REPLICATION_INPUT_SIZES:
            model = Encoder(32, 4)
            batch = np.zeros((4, 1, channels, samples), dtype=np.float32)
            batch = torch.from_numpy(batch)
            x_hat = model(batch)
            b, f, c, s = x_hat.shape

            self.assertEqual(b, 4)
            self.assertEqual(f, 1)
            self.assertEqual(s, samples // 2)
            self.assertEqual(c, channels)

    def test_Decoder(self):
        for channels, samples in REPLICATION_INPUT_SIZES:
            model = Decoder(32, 4)
            batch = np.zeros((4, 1, channels, samples // 2), dtype=np.float32)
            batch = torch.from_numpy(batch)
            x_hat = model(batch)
            b, f, c, s = x_hat.shape

            self.assertEqual(b, 4)
            self.assertEqual(f, 1)
            self.assertEqual(s, samples)
            self.assertEqual(c, channels)


class TestTimeConv(unittest.TestCase):
    KERNEL_SIZES = [2, 5]
    STRIDE_SIZES = [1, 3]
    DILATION_SIZES = [1, 3]
    L1_KERNELS = [16, 64]
    L2_KERNELS = [32, 128]
    L3_KERNELS = [32, 128]

    def test_TimeConvKernels(self):
        for l1 in TestTimeConv.L1_KERNELS:
            for l2 in TestTimeConv.L2_KERNELS:
                for l3 in TestTimeConv.L3_KERNELS:
                    for channels, samples in REPLICATION_INPUT_SIZES:
                        model = TimeConvModel(
                            channels,
                            samples,
                            l1_kernels=l1,
                            l2_kernels=l2,
                            l3_kernels=l3,
                        )
                        batch = np.zeros((4, 1, channels, samples), dtype=np.float32)
                        batch = torch.from_numpy(batch)
                        model(batch)

    def test_TimeConvBatchNorm(self):
        for bn in [True, False]:
            for channels, samples in REPLICATION_INPUT_SIZES:
                model = TimeConvModel(channels, samples, batch_norm=bn)
                batch = np.zeros((4, 1, channels, samples), dtype=np.float32)
                batch = torch.from_numpy(batch)
                model(batch)

    def test_TimeConvDilationSize(self):
        for d in TestTimeConv.DILATION_SIZES:
            for channels, samples in REPLICATION_INPUT_SIZES:
                model = TimeConvModel(channels, samples, dilation=d)
                batch = np.zeros((4, 1, channels, samples), dtype=np.float32)
                batch = torch.from_numpy(batch)
                model(batch)

    def test_TimeConvStrideSize(self):
        for s in TestTimeConv.STRIDE_SIZES:
            for channels, samples in REPLICATION_INPUT_SIZES:
                model = TimeConvModel(channels, samples, stride=s)
                batch = np.zeros((4, 1, channels, samples), dtype=np.float32)
                batch = torch.from_numpy(batch)
                model(batch)

    def test_TimeConvSizes(self):
        for d in TestTimeConv.DILATION_SIZES:
            for s in TestTimeConv.STRIDE_SIZES:
                for ks in TestTimeConv.KERNEL_SIZES:
                    for channels, samples in REPLICATION_INPUT_SIZES:
                        model = TimeConvModel(
                            channels, samples, kernel_size=ks, stride=s, dilation=d
                        )
                        batch = np.zeros((4, 1, channels, samples), dtype=np.float32)
                        batch = torch.from_numpy(batch)
                        model(batch)

    def test_TimeConvKernelSize(self):
        for ks in TestTimeConv.KERNEL_SIZES:
            for channels, samples in REPLICATION_INPUT_SIZES:
                model = TimeConvModel(channels, samples, kernel_size=ks)
                batch = np.zeros((4, 1, channels, samples), dtype=np.float32)
                batch = torch.from_numpy(batch)
                model(batch)


class TestModelForward(unittest.TestCase):
    def test_TimeResModel(self):
        for channels, samples in REPLICATION_INPUT_SIZES:
            model = TimeResModel(channels, samples)
            batch = np.zeros((4, 1, channels, samples), dtype=np.float32)
            batch = torch.from_numpy(batch)
            model(batch)

    def test_TransformerModel(self):
        for channels, samples in REPLICATION_INPUT_SIZES:
            model = TransformerModel(channels, samples)
            batch = np.zeros((4, 1, channels, samples), dtype=np.float32)
            batch = torch.from_numpy(batch)
            model(batch)

    def test_TimeConvModel(self):
        for channels, samples in REPLICATION_INPUT_SIZES:
            model = TimeConvModel(channels, samples)
            batch = np.zeros((4, 1, channels, samples), dtype=np.float32)
            batch = torch.from_numpy(batch)
            model(batch)

    def test_LfCnn(self):
        for channels, samples in REPLICATION_INPUT_SIZES:
            model = LfCnn(channels, samples)
            batch = np.zeros((4, 1, channels, samples), dtype=np.float32)
            batch = torch.from_numpy(batch)
            model(batch)

    def test_VarCnn(self):
        for channels, samples in REPLICATION_INPUT_SIZES:
            model = VarCnn(channels, samples)
            batch = np.zeros((4, 1, channels, samples), dtype=np.float32)
            batch = torch.from_numpy(batch)
            model(batch)


if __name__ == "__main__":
    unittest.main()
