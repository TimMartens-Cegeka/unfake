# Research: Establishing Detection Capabilities and Limits

Ticket: `.scratch/ai-photo-check/issues/01-detection-capabilities.md`
Researched: 2026-09-27

## Question

Which available indicators and detection methods can credibly assess fully AI-generated
images and post-capture AI edits, including enhancement and retouching? What evidence
supports each category, and what benchmark results, coverage gaps, and failure modes
require narrowing the MVP's claims or supported scope?

## How to read this report

Claims are marked **(verified)** when they come from a peer-reviewed paper, an official
spec, or an independent audit/measurement; **(vendor claim, unverified)** when they come
only from a vendor's own marketing/docs with no independent reproduction; and **(gap)**
when no credible source could be found at all. Dates are given wherever known. This
report favors depth on scope-limiting evidence over exhaustiveness, per the research
skill's brief.

---

## 1. Detecting fully AI-generated images

### 1.1 What works, and how well

- **CNN classifiers trained on one generator generalize to some unseen generators, but
  calibration is fragile.** Wang et al., *CNN-generated images are surprisingly easy to
  spot… for now* (CVPR 2020, arXiv:1912.11035, 2019-12-23): a ResNet-50 trained only on
  ProGAN with blur/JPEG augmentation reaches 92.6% mean AP across 11 unseen test sets,
  but per-class *accuracy* (as opposed to AP, i.e. once you pick an operating threshold)
  collapses to ~50–54% on some families (SAN, DeepFake) — high separability does not
  imply a usable public-facing accuracy number **(verified)**.
- **Frequency/spectral artifacts from GAN upsampling are a real, well-replicated signal**
  for GAN-era images: Durall et al. (arXiv:2003.01826, 2020), Frank et al. (arXiv:2003.08685,
  2020), Dzanic et al. (arXiv:1911.06465, NeurIPS 2020) all report 95–100% accuracy on
  GAN benchmarks of that era **(verified)**. This signal is architecture-specific
  (up-convolution/checkerboard artifacts) and does not transfer to diffusion models,
  which don't share that architecture (Corvi et al., arXiv:2211.00680, 2022-11-01).
- **Diffusion-specific detectors exist and outperform CNN baselines in-domain**, but are
  themselves narrow. DIRE (arXiv:2303.09295, ICCV 2023) gets 99.9% in-domain accuracy on
  its DiffusionForensics benchmark vs. 51.1% for the 2020 CNN baseline on the same suite —
  but DIRE's own cross-generator numbers and later papers show diffusion-vs-diffusion
  generalization is still incomplete, and GAN-trained detectors (Corvi's DMimageDetection,
  arXiv:2211.00680) are near chance (~51–52% accuracy) on unseen diffusion families like
  ADM/DALL·E2 even though AUC looks fine, and collapse further under social-media-style
  recompression **(verified)**.
- **CLIP-feature-based "universal" detectors are the current best generalizers.** Ojha et
  al.'s UnivFD (arXiv:2302.10174, CVPR 2023) — a frozen CLIP:ViT-L/14 + linear probe — hits
  93.4% mAP vs. 81.5% for the 2020 CNN baseline, with the gain concentrated on unseen
  diffusion/autoregressive generators. Cozzolino et al. (arXiv:2312.00195, 2023–2024)
  extend this to 18 generators including commercial ones (DALL·E 3: 86.3% AUC, Midjourney:
  92.9%, Adobe Firefly: 87.2%) but show that under "laundering" (crop+resize+JPEG) an
  un-augmented model's AUC drops to 44.7–65.6%, and even an augmented model only holds
  85.2% avg AUC vs. a clean-condition ~90–94% **(verified)**. This is the most credible
  present-day method family, and it still degrades meaningfully under realistic
  real-world image handling.

### 1.2 Benchmarks and generalization: the headline numbers do not hold up in the wild

- **GenImage** (arXiv:2306.08571, 2023-06-14): same-generator accuracy 92–99.9%, but
  cross-generator accuracy for a same-family-trained ResNet-50 drops to 52–55% (near
  chance) on unseen families (Midjourney, ADM, BigGAN). JPEG at q=30 drops accuracy to
  ~51.2%; 64px downscaling to ~57.4% for an unaugmented model **(verified)**.
- **Chameleon** (arXiv:2406.19435, 2024-06-27, updated 2025-02-15) is a hand-curated set
  of AI images that fool humans into believing they're real. Detectors that score 90–100%
  on older benchmarks (CNNSpot, DIRE, UnivFD, and the paper's own AIDE) score **0.08–5.04%
  fake-detection accuracy** on Chameleon while still reporting 92–100% on real images —
  i.e., models default to calling hard fakes "real." This is direct evidence that
  benchmark accuracy numbers substantially reflect benchmark-set overfitting, not
  generalizable capability **(verified)**.
- **A related paper (arXiv:2403.17608) argues a large share of GenImage's apparent
  accuracy is itself an artifact of JPEG/size biases baked into the dataset**, not a
  genuine generator signal — a caution against citing GenImage numbers uncritically
  **(verified, argued)**.
- **Screenshots/re-photographing destroy detector accuracy.** RRDataset (arXiv:2509.09172,
  2025-09-11) shows detector fake-accuracy roughly unchanged (~90%) under ordinary
  social-media-style transmission, but crashing to 0.05–1.42% after re-digitization
  (screen recapture/rephotograph) for DIRE/DNF; a more robust method (DRCT-ConvB) still
  falls from 93.5% to 64.3% **(verified)**.
- **Adversarial fragility is demonstrated against a real commercial product.** Mavali et
  al. (arXiv:2410.01574, 2024-10-02) drove ROC AUC as low as 4.1 (worse than random) on
  Stable Diffusion images using a cheap black-box attack requiring no knowledge of the
  detector's internals, and the attack survived social-media-style recompression; the
  attack succeeded against Hive's commercial detector, and Hive itself confirmed the
  result **(verified)**. Any public accuracy claim should be scoped to non-adversarial use.
- **False positives on ordinary real photos may be much higher than commonly assumed once
  detectors are recalibrated on modern content.** "LAION-Mobile" (arXiv:2609.11134) found
  that on modern AI-generated content, no detector tested exceeded AUC 0.624 (five of
  twelve scored below chance), and that thresholds tuned to legacy GAN data flag
  **17–91% of real smartphone photos** as AI-generated once recalibrated for current
  generators — attributed to modern computational-photography pipelines (multi-frame
  fusion, noise/motion-blur suppression) making real phone photos statistically resemble
  synthetic ones **(verified — single source, but directly relevant and alarming; flagged
  for independent replication before being relied on as a headline number)**.

### 1.3 Provenance/watermarking signals

- **C2PA (Coalition for Content Provenance and Authenticity)**, current spec v2.4
  (2026-04-01, spec.c2pa.org): a cryptographic manifest/assertion chain (hard bindings =
  content hash, tamper-evident but destroyed by re-encoding; soft bindings = perceptual
  hash/invisible watermark, meant to survive re-encoding and allow "durable" credential
  recovery). The spec itself acknowledges (§2.4.2) that an asset "can become separated
  from its C2PA Manifest due to removal or corruption of asset metadata" — i.e., ordinary
  metadata-based credentials are not stripping-resistant by design **(verified, primary
  spec)**. The spec's own §17.1 threat-model section is a stub deferring detailed security
  analysis to a future document **(verified, spec gap)**.
- **An independent academic security audit found C2PA does not meet its own claimed
  security goals.** Golaszewski et al., *Verifying Provenance of Digital Media: Why the
  C2PA Specifications Fall Short* (arXiv:2604.24890, 2026-04-27; also IACR ePrint
  2026/804) cites timestamp-tampering, weak/inconsistent revocation, and a weak
  conformance program, concluding C2PA "should not yet be relied upon for high-stakes
  uses such as financial disclosures, journalism, or legal evidence" **(verified,
  independent audit)**.
- **Google SynthID**: the peer-reviewed Nature paper (Dathathri et al., *Nature* 634,
  818–823, 2024-10-23) covers **SynthID for text only**, not images. For images, the
  primary source is a vendor-authored, non-peer-reviewed preprint (Gowal, Bunel et al.,
  arXiv:2510.09263, 2025-10-10), which claims production-scale watermarking of "over ten
  billion images and video frames" but only benchmarks an external variant against
  literature baselines — the flagship production system's robustness is **(vendor claim,
  unverified)**. No independent (non-Google) peer-reviewed evaluation of SynthID-for-images
  robustness against cropping/compression/screenshotting was found — an explicit **(gap)**.
  Google's own blog (2025-02-06, on Reimagine/Magic Editor) admits "edits… may be too
  small for SynthID to label and detect" — a first-party admission of a detection floor
  **(verified, self-admission)**.
- **OpenAI** embeds both C2PA metadata and a licensed SynthID watermark in
  DALL·E/ChatGPT/API images, and its own Help Center explicitly states its C2PA metadata
  "can sometimes be stripped out when files are edited or uploaded to some platforms"
  **(vendor claim, but self-critical)**.
- **Meta** states it labels AI content using C2PA/IPTC metadata plus its own markers, but
  explicitly admits "it's not yet possible to identify all AI-generated content, and
  there are ways that people can strip out invisible markers" (Meta Newsroom, 2024-02)
  **(vendor claim, self-admission)**. Meta also revised its labeling policy twice in 2024
  (2024-04-05 → 2024-07-01 → 2024-09-12) after minor Photoshop retouching was incorrectly
  triggering "Made with AI" labels — direct evidence that even a major vendor's own
  provenance-based labeling had a public false-positive problem on ordinary edits
  **(verified, documented policy history)**.
- **Metadata survival across platforms is poorly studied for C2PA specifically** (the
  standard is only ~2–4 years old), but classic Exif/IPTC-IIM/XMP stripping is
  well-documented: IPTC's own Photo Metadata Working Group tests found Facebook and
  Instagram strip essentially all embedded metadata, and Twitter/X serves downscaled
  images with metadata stripped **(verified, IPTC primary testing)**. A smaller,
  lower-tier study (Soni 2025) found Instagram/Messenger/Snapchat/WhatsApp "image mode"
  retained only ~16.67% of Exif fields **(lower confidence, single non-Scopus source)**.
  No rigorous study measuring C2PA-manifest-specific survival through real
  screenshot/re-upload pipelines was found — an explicit **(gap)**.

### 1.4 Commercial vendor tools (fully-synthetic detection)

- **Hive AI**: no published accuracy % on its own docs; independently tested (Zhao et
  al., arXiv:2402.03214, ACM CCS 2024) with zero false positives on a clean set and
  88.7% "ADSR" under Gaussian noise (best of three commercial tools tested), but
  successfully driven to worse-than-chance AUC by an adversarial attack that Hive itself
  confirmed (arXiv:2410.01574) **(mixed: some independent verification, also a
  demonstrated adversarial failure)**.
- **Sightengine**: broad documented generator coverage, 0% false positives on 15 real
  news photos in a NewsGuard journalistic audit (May 2026) **(vendor docs + one
  journalistic audit)**.
- **AI or Not / Optic**: vendor claims ~98.9% accuracy, undisclosed methodology
  **(vendor claim, unverified)**; independently, ADSR drops to ~52.6% (near chance) under
  Gaussian noise (arXiv:2402.03214); NewsGuard found a 6.67% false-positive rate on real
  news photos, and the vendor's own CEO acknowledged low image quality causes false
  positives.
- **Reality Defender, Sensity AI, Truepic, GPTZero's image feature**: essentially **no
  independent third-party evaluation exists** for any of these on image detection
  specifically — an explicit **(gap)** that should temper any "we use vendor X, which is
  N% accurate" claim.
- **Cross-vendor journalistic audit** (NewsGuard, May 2026) tested five tools (Hive,
  Sightengine, AI or Not, ZeroGPT, ScamAI) against 15 real news photos: collective false
  positive rate 13.3%, with ZeroGPT at 20% and ScamAI at 40%; vendor reps attributed this
  to unusual lighting/high contrast/blur and low resolution/compression common in
  real-world photojournalism **(independent journalistic measurement, small sample)**.
- **No standardized, independent, cross-vendor benchmark exists** for commercial APIs;
  two 2026 papers on the topic explicitly note vendors report scores "on entirely
  different scales and conventions," making apples-to-apples comparison currently
  impossible **(verified gap)**.

---

## 2. Detecting post-capture AI edits (inpainting, generative fill, enhancement, retouching)

This category is materially less mature than full-image-generation detection, and the
evidence base is thinner and newer.

### 2.1 Localized manipulation (splicing/inpainting) detectors

- **Error Level Analysis (ELA)** is a widely circulated heuristic with no rigorous
  peer-reviewed benchmark support; forensic literature treats it as unreliable for modern
  manipulations and it does not appear in current SOTA benchmarks **(no credible primary
  support found — should not be presented publicly as a real signal)**.
- **CNN/transformer localizers** (CAT-Net/CAT-Net v2, arXiv:2108.12947, IJCV 2022;
  PSCC-Net; TruFor, arXiv:2212.10957, CVPR 2023) achieve meaningful localization on
  classic splicing/copy-move and even some GAN-based local edits, with TruFor
  specifically noted to hold up where "most other methods fail catastrophically" on
  GAN-based local manipulation (OpenForensics) and diffusion-based local manipulation
  (CocoGlide) **(verified)**. TruFor's own robustness ablations show real Facebook/
  WhatsApp-style recompression widens the performance gap between methods sharply
  **(verified)**.
- **Modern diffusion-based inpainting structurally evades this entire method family.**
  DiffusionPrint (arXiv:2604.12443, 2026) explains that latent-diffusion inpainting
  decodes the *entire* image through a VAE, so all pixels — including untouched regions —
  are regenerated, which "destroys and renders ineffective the localized forensic traces
  that traditional image forgery localization methods depend on" **(verified, and this is
  the single most important architectural reason localized-edit detection is hard)**.
  FUSED (arXiv:2608.28302, 2026) independently confirms detectors largely key off global
  artifacts rather than the actual edited pixels: restoring authentic pixels outside the
  edited mask "reduces strong detectors to chance."
- **PRNU/sensor-noiseprint methods are a statistically weak, sample-size-hungry signal**
  for small edited regions, and become materially harder to use after any panorama
  stitching, multi-frame fusion, or heavy recompression (general PRNU forensic
  literature; corroborated by TruFor's use of noiseprint only as one auxiliary channel
  among several, not a standalone method).

### 2.2 AI upscaling / non-generative "AI enhancement" — an explicit literature gap

- Direct searches found **no peer-reviewed paper whose stated purpose is distinguishing
  a real photo that has been AI-upscaled/enhanced from a genuinely high-resolution
  capture** **(explicit gap)**. The closest related work is scoped to video (arXiv:2205.10406)
  or to *attributing which upscaler* was used among known candidates for
  visual-appeal prediction (arXiv:2502.14013) — not a real-vs-enhanced authenticity test.
- The strongest available evidence actually points the other way: **AI
  enhancement/restoration is documented in the literature primarily as an anti-forensic
  evasion technique**, not as something reliably detected. TGIF2 (arXiv:2603.28613)
  reports that "generative super-resolution significantly weakens forensic traces,
  demonstrating that common image enhancement operations can undermine current forensic
  pipelines." Tahir & Bal (arXiv:2405.02751, 2024-05-04) show AI restoration/denoising
  models can deliberately erase forensic traces used elsewhere in this report.
- Conclusion: **there is currently no credible basis for a public claim that this
  product can detect ordinary AI-based photo enhancement, upscaling, or "beauty filter"
  retouching.** This is a load-bearing finding for scoping the MVP.

### 2.3 Provenance signals for edits specifically

- C2PA's manifest structure does distinguish creation (`c2pa.created`, zero parent
  ingredients) from editing (`c2pa.opened` + `c2pa.ingredient` chain to a parent manifest)
  **(verified, primary spec)** — but this only works if every tool in the chain
  participates and the metadata survives transport, which (per §1.3 above) is not
  reliable, especially on social platforms.
- Adobe's own Photoshop documentation (2025-10-27) exposes only a UI on/off toggle for
  Content Credentials and does not document mask/region-level attribution for Generative
  Fill — Adobe's own granularity is "a generative-AI action occurred somewhere," not
  spatial **(verified, primary docs)**.
- Google's Magic Editor/Reimagine explicitly cannot label edits below some size
  threshold (§1.3 above) — a first-party admission that even the best-resourced
  in-house edit-provenance system has real coverage gaps.

### 2.4 Vendor tools claiming to detect edits/retouching specifically

- **No vendor's primary documentation was found clearly marketing a distinct product for
  detecting localized/partial AI edits** (as opposed to full synthesis). Sensity AI's own
  site has the strongest language, claiming its "File Forensic Analysis" detects
  "editing, tampering, or AI-generated content" via metadata/structural-consistency
  analysis — but this is **(vendor claim, unverified)** with no independent evaluation
  located. Illuminarty claims it identifies "which regions of the image have been
  generated" — the single strongest vendor claim found relevant to partial-edit
  detection, also **(vendor claim, unverified)**, and one independent informal test found
  a false positive on a real photo.
- Truepic is a capture-time provenance tool, not a post-hoc analyzer, and therefore has
  **no capability at all for the large majority of real-world images** that were never
  captured with a participating app/device.
- **No rigorous third-party benchmark was found testing any commercial tool's ability to
  distinguish a real photo with a localized AI edit from either an untouched photo or a
  fully synthetic one** — an explicit **(gap)** that itself argues against publicly
  claiming edit/retouch detection as a supported capability.

### 2.5 Failure modes specific to edit detection

- Localized edits are inherently harder than full synthesis because the manipulated
  region is diluted by surrounding authentic-image statistics, and modern diffusion
  inpainting regenerates the whole frame, erasing the boundary artifacts these methods
  rely on (§2.1).
- **Ordinary computational photography is a major false-positive source and is easy to
  confuse with "AI edited."** Apple's Deep Fusion (Apple Newsroom, 2019-09-10) performs
  ML-based multi-frame pixel fusion invisibly and by default in medium/low light; Google's
  HDR+ (Hasinoff et al., SIGGRAPH Asia 2016) states "every photo taken with HDR+ is
  actually a composite" — both alter noise statistics as standard, non-AI-edit behavior.
  A leading forensic-authentication vendor's own technical blog (Amped Software,
  updated 2026-04) names this as the field's "biggest authentication challenge," noting
  panorama stitching and multi-frame fusion make PRNU analysis "difficult, if not
  impossible" and that default HDR modes can trigger false suspicion of tampering with
  zero actual editing **(verified, though partly vendor-sourced)**.
- Compression/resizing (the norm for any social-media-uploaded photo) degrades the
  same double-JPEG-discontinuity and noise-residual signals these methods depend on;
  TruFor's own ablations and CAT-Net's compression-dependent design both illustrate this.
- **A joint government cybersecurity advisory** (NSA/CISA/ASD/CCCS, "Content Credentials:
  Strengthening Multimedia Integrity in the Generative AI Era," 2025-01-29 — content
  corroborated via secondary sources, primary PDF fetch blocked in this pass) reportedly
  states Content Credentials "do not guarantee media authenticity or full protection
  against sophisticated manipulation... should be used as one part of a comprehensive
  security strategy" **(reported, recommend re-verifying primary PDF before quoting
  publicly)**.

---

## 3. Cross-cutting conclusions relevant to MVP scope

1. **Fully-AI-generated detection is the more mature of the two categories, but still
   fails on: unseen/novel generators, recompressed or resized images, screenshots, and
   adversarially perturbed images — and independently-measured false-positive rates on
   ordinary real photos can be very high once thresholds are tuned to modern content.**
   Any headline accuracy number should be scoped ("on known generators, unmodified image,
   non-adversarial conditions") rather than presented as general-purpose.
2. **Post-capture AI edit/enhancement/retouching detection is substantially less mature.**
   Localized generative edits (inpainting/Generative Fill) are detectable in some cases
   with recent forensic localizers, but modern diffusion-based inpainting is specifically
   documented to defeat the mechanism these localizers rely on. Detecting non-generative
   AI enhancement/upscaling has **no supporting peer-reviewed literature at all** and
   should not be claimed.
3. **Provenance/metadata (C2PA, SynthID, platform-native labels) is a real, spec-backed
   signal but is not a substitute for content-based detection.** It only covers images
   whose full creation/edit chain participated, survives poorly through social-media
   re-upload and screenshotting by design, and even its own maintainers (OpenAI, Meta,
   Google, and an independent academic security audit of C2PA itself) publicly acknowledge
   it can be stripped or has coverage gaps.
4. **No commercial vendor's claimed accuracy is independently verified end-to-end.**
   Independent evaluation exists only for a handful of vendors (Hive, AI or Not/Optic,
   Illuminarty via one CCS 2024 paper; five vendors via one 2026 journalistic audit), and
   even those show real weaknesses (adversarial attacks succeeding against Hive; near-chance
   performance under noise for AI or Not). Reality Defender, Sensity, Truepic, and
   GPTZero's image feature have no independent evaluation found at all.
5. **Benchmark accuracy claims are prone to overfitting and should not be taken at face
   value.** Chameleon (2024) shows detectors scoring 90–100% on older benchmarks scoring
   0.08–5.04% on a harder, more realistic set of images humans mistake for real.

## Key scope recommendation for the MVP (for the parent to record)

Given the above, a public-facing product can credibly:
- Report **AI-signal indicators for likely-full-synthesis content** using
  CLIP-feature-style universal detectors and/or vendor APIs, but must present results as
  probabilistic evidence ("signals detected" / "no reliable signals" / "insufficient
  evidence"), never certification — consistent with the domain language already in
  `CONTEXT.md`.
- Check for and surface **provenance metadata (C2PA/Content Credentials, IPTC
  digitalSourceType) when present**, clearly labeled as a separate, complementary signal
  that is easily lost and whose absence proves nothing.
- **Should not claim** to reliably detect localized AI edits (Generative Fill/inpainting),
  and **should not claim at all** to detect non-generative AI enhancement, upscaling, or
  retouching — there is no credible evidence base to support either claim today, and
  ordinary computational photography (HDR/multi-frame fusion, standard on virtually all
  modern phone cameras) is a well-documented false-positive risk that would need to be
  explicitly excluded or heavily caveated if any "edited" claim is made at all.
- Should present any accuracy/confidence claims as bounded to known generators,
  non-adversarial conditions, and images that haven't been heavily recompressed,
  resized, or re-photographed/screenshotted — all of which are common in real-world
  public usage and are documented to collapse detector performance.

## Notable gaps for follow-up before finalizing public claims

- No independent, non-vendor-authored, peer-reviewed evaluation of Google SynthID's
  image-watermark robustness exists yet.
- No rigorous study measures C2PA-manifest (as opposed to plain Exif) survival rates
  across major social platforms.
- No independent benchmark tests any commercial vendor specifically on the
  full-synthesis-vs-localized-edit-vs-untouched three-way distinction this product needs.
- The NSA/CISA/ASD/CCCS Content Credentials advisory and CAT-Net's exact recompression
  F1 numbers should be re-fetched from primary sources before public quoting.

---

*Research conducted via two parallel research passes (fully-generated detection;
post-capture edit detection), each following the `research` skill's primary-source
protocol. Full per-claim citations (arXiv IDs, vendor URLs, dates) are given inline above;
see task transcripts for additional citation detail not reproduced here for brevity.*
