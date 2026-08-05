from physicalai.inference import InferenceModel
from physicalai.inference.runners import SinglePass
from physicalai.benchmark.gyms import LiberoBenchmark
from physicalai.policies.pi05 import Pi05


if __name__ == "__main__":
    # Load from HuggingFace (equivalent to your lerobot command)
    # policy = Pi05(
    #     pretrained_name_or_path="lerobot/pi05_libero_finetuned",
    #     compile_model=False,
    #     dtype="float32",       
    # )

    # policy.to("cuda")

    # benchmark = LiberoBenchmark(
    #     task_suite="libero_10",   # --env.task=libero_10
    #     task_ids=[6],             # --env.task_ids=[6] — reproduces libero_10_6
    #     num_episodes=5,          # --env.num_episodes=10
    #     max_steps=500,            # match the 500 steps seen in the failing run
    # )
    # results = benchmark.evaluate(policy)
    # print(results.summary())

    # policy = Pi05(
    #     pretrained_name_or_path="lerobot/pi05_libero_finetuned_v044",
    #     compile_model=False,
    # )
    # policy.to_openvino("pi05_libero_finetuned_hf_ov")
    
    model = InferenceModel(
        export_dir="exports/pi05_libero_runtime_ov",
        device="GPU",  # or "GPU" for Intel iGPU/dGPU
    )

    benchmark = LiberoBenchmark(
        task_suite="libero_10",
        max_steps=500,
        num_episodes=3,
    )
    results = benchmark.evaluate(model)
    print(results.summary())