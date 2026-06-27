# OpenClarity

> *Understanding Made Clear*

OpenClarity is an open-source AI visual assistance platform for blind and low-vision users. Built on smart glasses, it delivers hands-free, first-person visual information — reading text, identifying objects, and describing scenes — so users can navigate everyday life with greater independence and confidence.

---

## Mission

To give every blind and low-vision person clear, trustworthy, and affordable access to the visual information they need for everyday life.

We believe assistive technology should be:

- **Accessible** — priced for real people, not institutions
- **Transparent** — open source so anyone can inspect, verify, and improve it
- **Safe** — honest about what AI can and cannot reliably do

OpenClarity provides *information assistance, not mobility assistance*. We help users understand what is in front of them. We do not replace a cane, a guide dog, or human judgment.

---

## Vision

A world where open-source assistive technology is the standard — where blind and low-vision users everywhere can access professional-grade visual assistance without depending on expensive, opaque, closed systems.

We are building toward a complete open-source assistive technology ecosystem:

| Platform | Purpose |
|---|---|
| OpenClarity Vision | AI smart glasses (flagship) |
| OpenClarity Audio | Spatial audio assistance |
| OpenClarity Cane | Smart cane integration |
| OpenClarity Platform | Unified open SDK for developers |

Think of OpenClarity the way you think of Arduino or Linux: a trusted open foundation that anyone can build on.

---

## Values

### 1. Open by Default

Every line of our code is public. Our algorithms are auditable. Our roadmap is community-driven. We chose the name *Open*Clarity because openness is not a feature — it is the foundation.

We will never hide what our AI does, how it makes decisions, or what data it uses.

### 2. Clarity for Users

"Clarity" means understanding. For a blind user, clarity is not about vision — it is about knowing:

- What the label says
- Where the keys are
- What is in front of you
- Which button starts the machine

OpenClarity turns visual uncertainty into clear, concise, actionable information. One answer. Short sentences. No guessing.

### 3. Clarity in Code

Open source means nothing if the code is unreadable. We hold ourselves to the same standard we hold our product: transparent, clean, and understandable. Code clarity is a form of respect for the contributors and communities who depend on us.

### 4. Safety Honesty

We state clearly what OpenClarity is designed to do — and what it is not:

| We do | We do not |
|---|---|
| Read text (menus, labels, signs) | Navigate traffic or crosswalks |
| Identify objects and locations | Replace a guide dog or white cane |
| Describe scenes and environments | Guarantee obstacle-free paths |
| Guide multi-step tasks (appliances, forms) | Make medical, legal, or financial decisions |
| Provide scene summaries | Perform always-on facial recognition |

When AI confidence is low, we say so. Users deserve honesty over false reassurance.

### 5. Affordability as a Right

Current leading assistive glasses cost $1,700–$4,500. That price excludes the majority of the 285 million blind and low-vision people worldwide.

OpenClarity targets **$299–$499** for hardware by combining open-source software with accessible commodity hardware. We believe the price of independence should not depend on how wealthy you are.

### 6. Privacy by Design

Smart glasses with cameras and microphones are high-trust devices. We earn that trust through design:

- **Tap-to-capture** by default — the camera is never passively recording
- **No raw image storage** — images are not saved after processing
- **No background audio recording**
- **User-controlled history** — delete everything, anytime
- **Local processing first** — OCR and simple object recognition run on-device when possible

---

## What OpenClarity Does Today

OpenClarity focuses on four high-value, low-risk task categories:

### Read
Menus, medicine labels, mail, price tags, appliance buttons, packages, signs.

*"Read this."* → *"This is a bottle of Tylenol Extra Strength, 500mg. The label says: adults take 2 tablets every 6 hours."*

### Find
Keys, glasses, remotes, wallets, doors, cups — anything in your immediate environment.

*"Find my keys."* → *"Keys are on the table to your right, near the black wallet."*

### Describe
Your immediate surroundings, room layout, what is directly ahead.

*"What's in front of me?"* → *"You're facing a kitchen. Table in the center, fridge on the right, floor looks clear."*

### Ask
Open-ended visual questions about anything in your field of view.

*"How do I start this washing machine?"* → *"The Start button is the large button on the lower right. Current mode shows 'Normal'."*

---

## Technology Approach

OpenClarity runs on smart glasses with a first-person camera, bone-conduction audio, and voice input. This form factor reduces interaction friction that phone-based tools cannot eliminate:

**Phone:** unlock → open app → point camera → hold steady → wait → hear result

**OpenClarity glasses:** tap or say a word → look at target → hear result

Key architecture principles:
- **Local-first for speed** — OCR and common object detection run on-device
- **Cloud for complexity** — scene understanding and open-ended questions use VLM APIs
- **Streaming output** — short answer first, detail follows
- **Conservative mode for sensitive domains** — medical, legal, and financial content is read verbatim, never interpreted

---

## MVP Success Criteria

We define early success as:

- 80%+ of everyday low-risk questions receive a useful answer
- Severe errors (wrong medicine name, missed hazard) under 2%
- Average response time under 4 seconds
- Users prefer glasses over phone for at least 3 distinct task types
- Users choose to ask the AI first, before reaching for their phone or calling someone

---

## Project Structure

```
openclarity/
├── docs/          Documentation, research, product specs
├── product/       Product design and roadmaps
└── research/      Market research, user research, competitive analysis
```

---

## Roadmap

**Phase 1 — AI-only, non-critical tasks**
Read, Find, Describe, Ask on Halo smart glasses hardware.

**Phase 2 — AI + human fallback**
AI handles 70–80% of tasks. Low-confidence or high-stakes requests route to a human volunteer or trusted contact.

**Phase 3 — Personal context**
Remember where your medications are kept. Learn your home appliances. Recall your regular bus stop. Build a model of your daily environment.

**Phase 4 — Specialized skill modules**
Medication management, kitchen assistance, shopping, mail reading, appliance operation, indoor object location.

---

## Get Involved

OpenClarity is community-driven. Whether you are a developer, a blind or low-vision user, a researcher, or an accessibility advocate — there is a place for you here.

- **GitHub:** [github.com/openclarity](https://github.com/openclarity)

---

*OpenClarity — Open Source, Crystal Clear.*
