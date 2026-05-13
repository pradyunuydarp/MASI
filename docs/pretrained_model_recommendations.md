# Transfer Learning Choices For MASI

Searched: 2026-05-12

This table chooses one practical transfer-learning starting point for each MASI component where transfer is appropriate. If local training budget is limited, the best external-transfer compromise for Phase 3 is to use a pretrained text-to-text Transformer over serialized MASI semantic-ID strings.

| MASI model/component | Decision | Selected model/checkpoint | Why this fit |
|---|---|---|---|
| Text encoder for product metadata | Transfer learning | [`Marqo/marqo-fashionSigLIP`](https://huggingface.co/Marqo/marqo-fashionSigLIP) | Best fit for the current Clothing/Shoes/Jewelry domain because it is a fashion/e-commerce multimodal embedding model. Use it as the frozen text encoder replacement for current `openai/clip-vit-base-patch32`. |
| Vision encoder for product images | Transfer learning | [`Marqo/marqo-fashionSigLIP`](https://huggingface.co/Marqo/marqo-fashionSigLIP) | Same model should be used for images so text and vision embeddings stay in the same pretrained multimodal space. |
| Text projection head | Train locally/scratch | Train MASI projection head | The head learns behavior-aware structure from the local CSJ user-item graph, so an external pretrained projection head would not encode our collaborative positives/negatives. |
| Vision projection head | Train locally/scratch | Train MASI projection head | Same reason as the text head: this is a behavior-alignment adapter over the chosen frozen vision embedding space. |
| Behavior-aware alignment model | Train locally/scratch | Train MASI alignment model | The transferable part is the frozen encoder. The alignment objective depends on local user histories, graph-hard negatives, and interaction windows. |
| Text RQ-VAE encoder/decoder/codebooks | Train locally/scratch | Train MASI text RQ-VAE | Semantic-ID codebooks are tied to the exact embedding distribution and item set. Closest reference found: [`edobotta/rqvae-amazon-beauty`](https://huggingface.co/edobotta/rqvae-amazon-beauty), but it is Amazon Beauty/text-only and not a good checkpoint transfer target for CSJ MASI. |
| Vision RQ-VAE encoder/decoder/codebooks | Train locally/scratch | Train MASI vision RQ-VAE | No suitable visual RQ-VAE checkpoint was found. The visual codebook should be learned from the selected CSJ item-image embeddings. |
| Late-fused semantic-ID generator | Train locally/scratch | Generate from MASI text and vision RQ-VAEs | This is deterministic artifact construction from local codebooks, not a pretrained neural component. |
| Cross-modal MLM Transformer | Transfer learning | [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small), or `google-t5/t5-base` if memory allows | Best available external checkpoint for masked/span reconstruction because T5 was pretrained with text-to-text denoising. Serialize MASI IDs as text, such as `[TXT] txt_c0_17 txt_c1_4 [VIS] <extra_id_0>`, and fine-tune lightly. This is more appropriate than BERT if the same model will later generate item IDs. |
| Autoregressive generative SID recommender | Transfer learning | [`google-t5/t5-small`](https://huggingface.co/google-t5/t5-small), preferably after the MASI cross-modal T5 fine-tune | T5 is the best practical external starting point for generative recommendation over artificial SID strings. It does not know MASI item IDs out of the box, so constrained decoding or light fine-tuning on user-history-to-next-SID examples is still needed. |
| SASRec-style sequential baseline | Transfer-learning reference | [`RUCAIBox/UniSRec`](https://github.com/RUCAIBox/UniSRec) | UniSRec is the closest pretrained sequential-recommendation transfer candidate. It is best treated as a baseline/adaptation reference, not as a replacement for MASI's generative SID recommender. |
| Full generative-recommendation reference | Transfer-learning reference only | [`snap-research/GRID`](https://github.com/snap-research/GRID) | GRID is a strong semantic-ID generative recommendation reference framework, but direct checkpoint transfer is not appropriate unless its tokenizer, item universe, SID vocabulary, and split protocol are aligned with MASI. |

## Priority Order

1. Replace the frozen CLIP encoder with `Marqo/marqo-fashionSigLIP`.
2. Use `google-t5/t5-small` for Phase 3 if we cannot afford training the custom MLM/recommender from scratch.
3. Fine-tune T5 first on cross-modal reconstruction, then on user-history-to-next-SID generation.
4. Keep projection heads and RQ-VAE/codebook learning local because those parameters are tied to the selected item set and generated embeddings.
5. Consider `RUCAIBox/UniSRec` only as a transfer-learning baseline/reference.
6. Use `snap-research/GRID` as an architecture/reference framework, not as a direct checkpoint source.

## Implementation Notes

- `Marqo/marqo-fashionSigLIP` will likely require changing the feature-extraction path from `CLIPModel`/`CLIPProcessor` to `AutoModel`/`AutoProcessor` with `trust_remote_code=True`.
- Do not reuse old MASI Phase 1, Phase 2, or Phase 3 checkpoints after changing the frozen encoder. The embedding space and downstream codebooks will change.
- If broader Amazon product coverage becomes more important than fashion specificity, the next encoder candidate is [`Marqo/marqo-ecommerce-embeddings-B`](https://huggingface.co/Marqo/marqo-ecommerce-embeddings-B).
- Using T5 for Phase 3 is a design change from the current custom PyTorch Transformer. It should be documented as a compute-saving transfer-learning variant.
- T5 will not understand MASI semantic-ID tokens without some adaptation. At minimum, add MASI special tokens/code tokens to the tokenizer or serialize them consistently as text and fine-tune with constrained outputs.
