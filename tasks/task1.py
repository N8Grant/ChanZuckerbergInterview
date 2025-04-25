from time import perf_counter

from cellpose import models

from chanzuck.utils.dataloader import CellposeZarrLoader


def run_benchmark(
    dataset_path: str,
    channel_indices: list[int],
    num_samples: int = 5,
    use_gpu: bool = True,
):
    print(f"\n🚀 Benchmarking Cellpose with GPU={'✅' if use_gpu else '❌'}")
    print(f"📁 Dataset: {dataset_path}")
    print(f"🔬 Channels: {channel_indices}  |  Samples: {num_samples}")

    loader = CellposeZarrLoader(dataset_path, channel_indices=channel_indices)
    model = models.Cellpose(gpu=use_gpu, model_type="nuclei")

    load_times = []
    infer_times = []

    for i in range(min(len(loader), num_samples)):
        try:
            # ⏱️ Load sample
            start = perf_counter()
            sample = loader[i]
            image = sample["image"]
            load_times.append(perf_counter() - start)

            # ⏱️ Inference
            start = perf_counter()
            _ = model.eval(image, channels=[0, None], z_axis=1, do_3D=True)
            infer_times.append(perf_counter() - start)

            print(
                f"  ✅ Sample {i+1:02d}  |  Load: {load_times[-1]:.4f}s  |  Infer: {infer_times[-1]:.4f}s"
            )

        except Exception as e:
            print(f"  ⚠️ Sample {i+1:02d} failed: {e}")
            continue

    # 📊 Summary
    if load_times and infer_times:
        print("\n📊 Benchmark Summary:")
        print(
            f"  📂 Avg Load Time     : {sum(load_times)/len(load_times):.4f} sec"
        )
        print(
            f"  🧠 Avg Inference Time: {sum(infer_times)/len(infer_times):.4f} sec"
        )
    else:
        print("❌ No successful samples to report.")


# 🔁 Run both GPU and CPU benchmarks
dataset_path = "data/20241107_infection.zarr"
channel_indices = [1]  # Set to DAPI channel or whichever you want

run_benchmark(dataset_path, channel_indices, use_gpu=True)
run_benchmark(dataset_path, channel_indices, use_gpu=False)
