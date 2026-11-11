# 参考文献（成员 1 维护）

> 格式：GB/T 7714 风格 + arXiv/DOI 链接。报告正文用 `[n]` 引用；新增文献请追加编号，不要重排。

[1] RADFORD A, KIM J W, HALLACY C, et al. Learning transferable visual models from natural language supervision[C]//Proceedings of the 38th International Conference on Machine Learning (ICML). 2021: 8748-8763. arXiv:2103.00020.

[2] JIA C, YANG Y, XIA Y, et al. Scaling up visual and vision-language representation learning with noisy text supervision[C]//ICML. 2021: 4904-4916. arXiv:2102.05918.

[3] ILHARCO G, WORTSMAN M, GADRE S, et al. OpenCLIP[CP/OL]. 2021. https://github.com/mlfoundations/open_clip.

[4] SCHUHMANN C, BEAUMONT R, VENCU R, et al. LAION-5B: An open large-scale dataset for training next generation image-text models[C]//NeurIPS Datasets and Benchmarks. 2022. arXiv:2210.08402.

[5] THRUSH T, JIANG R, BARTOLO M, et al. Winoground: Probing vision and language models for visio-linguistic compositionality[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022: 5238-5248. arXiv:2204.03162.

[6] YUKSEKGONUL M, BIANCHI F, KALLURI P, et al. When and why vision-language models behave like bags-of-words, and what to do about it?[C]//International Conference on Learning Representations (ICLR). 2023. arXiv:2210.01936.

[7] HSIEH C Y, ZHANG J, MA Z, et al. SugarCrepe: Fixing hackable benchmarks for vision-language compositionality[C]//NeurIPS Datasets and Benchmarks. 2023. arXiv:2306.14610.

[8] DUMPALA S H, JAISWAL A, SASTRY C, et al. SugarCrepe++ dataset: Vision-language model sensitivity to semantic and lexical alterations[J]. arXiv preprint arXiv:2406.11171, 2024.

[9] GOEL S, BAHL S, BHATIA S, et al. CyCLIP: Cyclic contrastive language-image pretraining[C]//NeurIPS. 2022. arXiv:2205.14459.

[10] DOVEH S, ARBELLE A, HARARY S, et al. Teaching CLIP to count to ten[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2023: 3170-3180. arXiv:2302.12066.

[11] LI J, LI D, XIONG C, et al. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation[C]//ICML. 2022: 12888-12900. arXiv:2201.12086.

[12] LI J, LI D, SAITO S, et al. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models[C]//ICML. 2023: 19730-19742. arXiv:2301.12597.

[13] YU J, WANG Z, VASUDEVAN V, et al. CoCa: Contrastive captioners are image-text foundation models[J]. Transactions on Machine Learning Research, 2022. arXiv:2205.01917.

[14] KRISHNA R, ZHU Y, GROTH O, et al. Visual Genome: Connecting language and vision using crowdsourced dense image annotations[J]. International Journal of Computer Vision, 2017, 123(1): 32-73.

[15] LIN T Y, MAIRE M, BELONGIE S, et al. Microsoft COCO: Common objects in context[C]//European Conference on Computer Vision (ECCV). 2014: 740-755.

[16] PLUMMER B A, WANG L, CERVANTES C M, et al. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models[C]//ICCV. 2015: 2641-2649.

[17] HONNIBAL M, MONTANI I. spaCy: Industrial-strength natural language processing in Python[CP/OL]. 2017. https://spacy.io.

[18] EFRON B. Bootstrap methods: Another look at the jackknife[J]. The Annals of Statistics, 1979, 7(1): 1-26.

[19] HU E J, SHEN Y, WALLIS P, et al. LoRA: Low-rank adaptation of large language models[C]//ICLR. 2022. arXiv:2106.09685.

[20] HINTON G, VINYALS O, DEAN J. Distilling the knowledge in a neural network[J]. arXiv preprint arXiv:1503.02531, 2015.

[21] ZHANG L, RAO A, AGRAWALA M. Adding conditional control to text-to-image diffusion models[C]//ICCV. 2023. arXiv:2302.05543. （仅方法附录引用，正文不展开）

[22] KIM W, SON B, KIM I. Vilt: Vision-and-language transformer without convolution or region supervision[C]//ICML. 2021: 5583-5594. arXiv:2102.03334.

[23] BUGLIARELLO E, RODOLA E, CULURCIELLO E. Is CLIP the main roadblock for fine-grained open-world perception?[J]. arXiv preprint arXiv:2407.03503, 2024.

[24] WANG Z, WU Z, AGARWAL D, et al. Vanilla CLIP knows more than what it says: On the compositionality of CLIP[C]//ICLR. 2025. arXiv:2410.22909.（如最终版引用，请核对会议信息）

## 数据与工具引用（报告附录用）

- Flickr30k：Young et al., "From image descriptions to visual denotations", TACL 2014.
- ARO 代码：https://github.com/mertyg/vision-language-models-are-bows （MIT License）。
- SugarCrepe++ 数据镜像：https://huggingface.co/datasets/Mayfull/sugarcrepepp_vlms
- Winoground 数据镜像：https://huggingface.co/datasets/mm-eval/Winoground
- OpenCLIP 权重：`ViT-B-32` / `laion2b_s34b_b79k`（LAION-5B 子集训练）。
