def main():
    import os
    import sys
    from pathlib import Path
    import torch
    
    project_root = Path(__file__).resolve().parents[3]
    os.environ["HF_HOME"] = str(project_root / "etc" / "smolvla_cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    lerobot_src = project_root / "etc" / "third_party" / "lerobot" / "src"
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    
    ckpt_path = project_root / "artifacts" / "Delta_Weights"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        print(f"Loading policy from {ckpt_path} to {device}...")
        policy = SmolVLAPolicy.from_pretrained(ckpt_path, local_files_only=True).to(device)
        policy.eval()
        
        print("Loading pre/post processors...")
        pre, post = make_pre_post_processors(
            policy.config, pretrained_path=ckpt_path,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}}
        )
        
        print("STRICT LOCAL LOAD = VERIFIED")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("STRICT LOCAL LOAD = FAILED")


if __name__ == "__main__":
    main()
