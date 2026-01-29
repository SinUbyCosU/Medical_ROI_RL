import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor


class RewardMLP(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(dim * 2)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, img_embeds: torch.Tensor, txt_embeds: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img_embeds, txt_embeds], dim=-1)
        x = self.norm(x)
        return self.net(x).squeeze(-1)


class RewardWrapper(nn.Module):
    """Wraps the Gemini-aligned reward model for DDPO."""

    def __init__(self, reward_model_path: str, device: str = "cuda", clip_model: str = "openai/clip-vit-base-patch16") -> None:
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.clip = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model)

        ckpt = torch.load(reward_model_path, map_location=self.device)
        dim = ckpt.get("dim")
        if dim is None:
            dim = getattr(self.clip.config, "projection_dim", 512)
        self.reward = RewardMLP(dim=dim)
        self.reward.load_state_dict(ckpt["model"])
        self.reward.to(self.device)
        self.reward.eval()

        self.clip = CLIPModel.from_pretrained(clip_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model)

    def forward(self, images, prompts, metadata=None):
        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors="pt", truncation=True).to(
            self.device
        )
        with torch.no_grad():
            img_embeds = self.clip.get_image_features(pixel_values=inputs["pixel_values"])
            txt_embeds = self.clip.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

            img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            txt_embeds = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

            scores = self.reward(img_embeds, txt_embeds)
        return scores
