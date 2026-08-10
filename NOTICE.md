# NOTICE

## Powered by MiniMax H3

This project generates video using **MiniMax H3**, which is **not** part of this
repository and is **not** distributed with it. You obtain the weights yourself
from [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and are
bound by their licence when you do.

> MiniMax H3 is licensed under the MiniMax H3 Community License Agreement,
> Copyright (c) 2026 MiniMax. All Rights Reserved.

Read the full text before running anything:
- [Licence](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [Licence Q&A](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/QA-about-License.md)

---

## Territory restriction -- read this first

**The MiniMax H3 Community License Agreement does not grant rights in the EU, the
UK, South Korea, or the USA.** These are the "Excluded Territories" (§I.5), and
the licence applies only outside them (§I.3).

The part people miss: **§V.4 restricts the Outputs, not just the weights.** You
may not use, reproduce, modify, distribute *or display* H3 or anything it
generates outside the Applicable Territory, and Exhibit A §1 lists use outside
the territory as a prohibited use outright. Running the container in a permitted
region does **not** cure this if you, or your audience, are in an excluded one --
the restriction attaches to the Licensee (§I.9) and to the Outputs.

This is not legal advice. If you are in an excluded territory, MiniMax provides a
route rather than a wall: **[apply for a licence](https://platform.minimax.io/h3-license)**.
Their own Q&A describes the limit as "not yet", not "not ever".

---

## Obligations that apply even for personal, non-commercial use

**Disclose AI-generated content.** Exhibit A §12 requires clearly and prominently
disclosing that content is machine-generated when publishing it in a public
environment. YouTube is such an environment and has its own altered/synthetic
content setting -- tick it. §III.3.b also encourages embedding an AI-generation
identifier in the file.

**Do not strip provenance.** This project deliberately contains no metadata
scrubbing. Removing attribution to disguise that output came from H3 would breach
the licence, and for a commercial product §IV.2 goes further and *requires*
displaying "MiniMax H3" prominently in the interface.

**Additional terms if this stops being personal:**
- §IV.2 -- commercial products must prominently display "MiniMax H3"
- §IV.1 -- over USD 20M annual revenue needs prior written authorisation
- §V.2 -- you must bind downstream users to equivalent restrictions and notify them
- §V.5 -- you must maintain safeguards against infringing outputs
- §V.3 -- outputs may not be used to improve any other AI model

Governing law is Hong Kong (§IX).

---

## Third-party components

| Component | Role | Licence |
|---|---|---|
| [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | video + audio generation | MiniMax H3 Community License |
| Qwen3-VL-32B (H3's encoder) | text/vision conditioning | Apache 2.0 |
| [diffusers](https://github.com/huggingface/diffusers) | pipeline runtime | Apache 2.0 |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | optional super-resolution | BSD-3-Clause |
| [spandrel](https://github.com/chaiNNer-org/spandrel) | loads upscaler checkpoints | MIT |
| [FFmpeg](https://ffmpeg.org/) | interpolation fallback, encoding | LGPL/GPL depending on build |

The code in this repository is MIT licensed (see `LICENSE`). That licence covers
this code only -- it does not and cannot relicense H3 or its outputs.
