# Article metadata

Fields match the `Article` entity in bandarrame_rs (`lib/src/entities/article.rs`).

- **Title:** Should I Have Trained the Model After All?
- **Slug:** `jev-vs-custom-classifier`
- **URL:** https://bandarra.me/posts/jev-vs-custom-classifier
- **Description:** In my last post, I skipped training a spam classifier and used Jev instead. Here, I compare Jev with a neural network and k-NN built from thousands of labeled comments in the Jigsaw toxicity dataset: how accurate each one is, how many labels they need, and what they cost to run.
- **Tags:** `jev`, `system-one`, `decision-model`, `classification`, `machine-learning`, `pytorch`, `vector-search` (all already used on the blog)
- **Hero image:** `hero-two-bouncers.png` (not generated yet, prompt below), served as `/images/hero-two-bouncers.png`. Alt text: "A pixel-art nightclub with two doors. At the left door, a bouncer with a "JEV" badge glances at a small card with a single rule on it. At the right door, a second bouncer checks each guest against a towering stack of photo binders. At both doors, smiling blue characters are let in while grumpy orange ones are held behind a velvet rope."
- **AI assisted:** yes
- **Status:** Draft

## Images used in the article
| File | Where |
|---|---|
| `hero-two-bouncers.png` | Hero, top of the article (to generate) |
| `ranking-quality.png` | Figure 1 |
| `jev-scores.png` | Figure 2 |
| `jev-threshold.png` | Figure 3 |
| `per-label.png` | Figure 4 |
| `labels-needed.png` | Figure 5 |
| `embedding-vs-classifier.png` | Figure 6 |

## Alternative titles
- Jev vs. a Custom Classifier: How Many Labels Does It Take?
- Describing a Classifier vs. Training One

## Hero image prompt

A sequel to the jev-comment-spam hero (`jev-comment-spam-hero.png`). Generate at a wide ratio and crop to 1344x574 like the first one. Pass the first hero as a style reference if the tool supports it.

> Wide 16-bit pixel art scene, crisp pixels, no anti-aliasing, retro video game style. The entrance of a nightclub at night: dark maroon brick wall covered in swirling magenta neon tubes, glossy dark blue tiled floor with soft reflections. The wall has TWO arched doors side by side, each glowing with warm yellow light from inside, each with gold stanchions and a red velvet rope.
>
> Left door: a bouncer in a black suit, white shirt, black tie, black sunglasses and slicked black hair, with a small white name badge that reads "JEV". He holds up one hand to stop a guest and calmly glances at a small white index card in his other hand with one short line written on it. Relaxed, confident posture.
>
> Right door: a second bouncer in a dark grey suit and sunglasses, no name badge, standing behind a tall wobbly tower of thick binders and stacks of photographs that reaches above his head. He holds a magnifying glass and compares a photo against the face of the guest in front of him. Busy, a little overwhelmed, but competent.
>
> Guests: round blue blob characters with big happy smiles walk through both doors into the light, waving. Round orange blob characters with grumpy faces, some wearing sunglasses, are held behind the velvet ropes at both doors, one with angry steam puffs above its head.
>
> Mood: playful, symmetrical composition, both doors equally busy. No other text in the image besides the "JEV" badge.
