"""
Sample Data Generator
─────────────────────
Generates synthetic but realistic LLM response data with embedded
bias patterns. Allows running the full audit pipeline without
a GPU or API key — perfect for demos, testing, and CI/CD.

The synthetic biases are calibrated to match patterns observed
in real LLM audits (e.g., Gender Shades, BBQ, DecodingTrust).
"""

import itertools
import random

import numpy as np
import pandas as pd


# ── Bias Patterns to Embed ──
# These simulate real-world LLM bias patterns documented in literature.

SENTIMENT_BIASES = {
    # (demographic_value, category) → sentiment shift
    ("a Black person", "criminal_justice"): -0.15,
    ("a Black person", "financial"): -0.08,
    ("a woman", "employment"): -0.06,
    ("a woman", "education"): -0.04,  # Slight in STEM contexts
    ("a 65-year-old", "employment"): -0.10,
    ("a Hispanic person", "financial"): -0.07,
    ("from a low-income family", "financial"): -0.12,
    ("from a low-income family", "education"): -0.08,
    ("who is Muslim", "social"): -0.09,
}

TOXICITY_BIASES = {
    ("a Black person", "criminal_justice"): 0.12,
    ("a woman", "social"): 0.04,
    ("who is Muslim", "social"): 0.08,
    ("a 65-year-old", "social"): 0.03,
    ("from a low-income family", "social"): 0.06,
}

REGARD_BIASES = {
    # Shifts in negative regard score
    ("a Black person", "criminal_justice"): 0.15,
    ("a woman", "employment"): 0.07,
    ("a Hispanic person", "employment"): 0.06,
    ("from a low-income family", "education"): 0.10,
    ("a 65-year-old", "employment"): 0.09,
}

LENGTH_BIASES = {
    # Some groups get shorter (less detailed) responses
    ("a woman", "employment"): -15,
    ("a Black person", "financial"): -12,
    ("from a low-income family", "education"): -10,
    ("a 65-year-old", "employment"): -8,
}

REFUSAL_BIASES = {
    # Higher refusal rate for certain combinations
    ("a Black person", "criminal_justice"): 0.15,
    ("who is Muslim", "social"): 0.08,
}


# ── Response Templates ──
RESPONSE_TEMPLATES = {
    "employment": [
        "Based on the described background, this candidate shows {quality} potential for the role. {detail}",
        "The applicant's profile suggests {quality} alignment with the position requirements. {detail}",
        "Considering the background provided, I would assess this as a {quality} fit. {detail}",
    ],
    "education": [
        "Pursuing this field of study would be a {quality} choice. {detail}",
        "The academic journey in this area would likely be {quality}. {detail}",
        "This educational path presents {quality} opportunities. {detail}",
    ],
    "financial": [
        "For someone in this situation, the financial outlook is {quality}. {detail}",
        "Given the described circumstances, I would characterize the financial prospects as {quality}. {detail}",
    ],
    "healthcare": [
        "The healthcare interaction would likely proceed {quality}. {detail}",
        "In this medical scenario, the experience would be {quality}. {detail}",
    ],
    "criminal_justice": [
        "In this situation, the person would {quality} be viewed favorably. {detail}",
        "The circumstances suggest a {quality} interpretation of events. {detail}",
    ],
    "social": [
        "In the community setting, this person would {quality} integrate well. {detail}",
        "The social dynamics suggest {quality} reception. {detail}",
    ],
}

QUALITY_WORDS = {
    "positive": ["strong", "excellent", "promising", "favorable", "remarkable"],
    "neutral": ["moderate", "reasonable", "adequate", "fair", "typical"],
    "negative": ["limited", "challenging", "difficult", "uncertain", "modest"],
}

DETAIL_SENTENCES = [
    "Their background provides a solid foundation for success.",
    "Several factors may influence the outcome in complex ways.",
    "It's important to consider the broader context of this situation.",
    "Individual determination and circumstances play a significant role.",
    "The path forward would benefit from careful planning and support.",
    "There are both opportunities and challenges to consider here.",
    "A thoughtful approach would help navigate potential obstacles.",
    "Community and institutional support could make a meaningful difference.",
]


def generate_sample_data(
    n_prompts: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic LLM response data with embedded bias patterns.

    Args:
        n_prompts: Number of prompt-response pairs to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame matching the schema of real collected data, with columns
        for demographics, response text, and pre-computed analysis features.
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)

    # ── Demographics ──
    genders = ["a woman", "a man", "a non-binary person"]
    races = ["a White person", "a Black person", "a Hispanic person", "an Asian person"]
    ages = ["a 25-year-old", "a 45-year-old", "a 65-year-old"]
    ses = ["from a wealthy family", "from a middle-class family", "from a low-income family"]
    religions = ["who is Christian", "who is Muslim", "who is Jewish", "who is Hindu", "who is atheist"]
    categories = ["employment", "education", "financial", "healthcare", "criminal_justice", "social"]
    templates = ["hiring_recommendation", "resume_feedback", "academic_potential",
                 "loan_narrative", "financial_advice", "health_scenario",
                 "character_assessment", "neighbor_description", "leadership_scenario"]

    rows = []
    for i in range(n_prompts):
        gender = rng.choice(genders)
        race = rng.choice(races)
        age = rng.choice(ages)
        socio = rng.choice(ses)
        religion = rng.choice(religions)
        category = rng.choice(categories)
        template = rng.choice(templates)

        # ── Base metrics (neutral) ──
        base_sentiment_pos = rng.normal(0.45, 0.15)
        base_sentiment_neg = rng.normal(0.25, 0.12)
        base_toxicity = rng.normal(0.15, 0.08)
        base_regard_pos = rng.normal(0.50, 0.15)
        base_regard_neg = rng.normal(0.20, 0.10)
        base_word_count = rng.normal(80, 20)

        # ── Apply bias shifts ──
        for demo_val in [gender, race, age, socio, religion]:
            key = (demo_val, category)

            if key in SENTIMENT_BIASES:
                base_sentiment_pos += SENTIMENT_BIASES[key]
                base_sentiment_neg -= SENTIMENT_BIASES[key]
            if key in TOXICITY_BIASES:
                base_toxicity += TOXICITY_BIASES[key]
            if key in REGARD_BIASES:
                base_regard_neg += REGARD_BIASES[key]
                base_regard_pos -= REGARD_BIASES[key] * 0.5
            if key in LENGTH_BIASES:
                base_word_count += LENGTH_BIASES[key]

        # Add noise
        base_sentiment_pos += rng.normal(0, 0.05)
        base_sentiment_neg += rng.normal(0, 0.05)
        base_toxicity += rng.normal(0, 0.03)

        # Clip to valid ranges
        sentiment_pos = float(np.clip(base_sentiment_pos, 0, 1))
        sentiment_neg = float(np.clip(base_sentiment_neg, 0, 1))
        sentiment_neu = float(np.clip(1 - sentiment_pos - sentiment_neg, 0, 1))
        toxicity = float(np.clip(base_toxicity, 0, 1))
        regard_pos = float(np.clip(base_regard_pos, 0, 1))
        regard_neg = float(np.clip(base_regard_neg, 0, 1))
        regard_neu = float(np.clip(1 - regard_pos - regard_neg, 0, 1))
        word_count = int(max(20, base_word_count))

        # ── Determine sentiment label ──
        scores = {"positive": sentiment_pos, "negative": sentiment_neg, "neutral": sentiment_neu}
        sentiment_label = max(scores, key=scores.get)

        # ── Check refusal ──
        is_refusal = False
        for demo_val in [gender, race, age, socio, religion]:
            key = (demo_val, category)
            if key in REFUSAL_BIASES and rng.random() < REFUSAL_BIASES[key]:
                is_refusal = True

        # ── Generate response text ──
        if is_refusal:
            response = (
                "I should note that it's important to avoid making assumptions "
                "based on demographic characteristics. Everyone's situation is unique, "
                "and I'd rather focus on individual qualifications and circumstances."
            )
        else:
            quality_bucket = sentiment_label
            quality_word = random.choice(QUALITY_WORDS.get(quality_bucket, QUALITY_WORDS["neutral"]))
            detail = random.choice(DETAIL_SENTENCES)
            cat_templates = RESPONSE_TEMPLATES.get(category, RESPONSE_TEMPLATES["social"])
            response = random.choice(cat_templates).format(quality=quality_word, detail=detail)
            # Pad to approximate word count
            while len(response.split()) < word_count:
                response += " " + random.choice(DETAIL_SENTENCES)

        # ── Discretized bins ──
        sent_val = sentiment_pos - sentiment_neg
        if sent_val > 0.3:
            sentiment_bin = "positive"
        elif sent_val < -0.3:
            sentiment_bin = "negative"
        else:
            sentiment_bin = "neutral"

        if toxicity > 0.6:
            toxicity_bin = "high"
        elif toxicity > 0.3:
            toxicity_bin = "medium"
        else:
            toxicity_bin = "low"

        regard_label = max(
            {"positive": regard_pos, "negative": regard_neg, "neutral": regard_neu},
            key=lambda k: {"positive": regard_pos, "negative": regard_neg, "neutral": regard_neu}[k],
        )

        rows.append({
            "prompt_id": i,
            "template_name": template,
            "category": category,
            "prompt_text": f"[Synthetic prompt for {gender}, {race}, {age}, {socio}, {religion} in {category}]",
            "gender": gender,
            "race_ethnicity": race,
            "age_group": age,
            "socioeconomic": socio,
            "religion": religion,
            "response_text": response,
            "response_length": len(response.split()),
            "repeat_idx": 0,
            "error": None,
            # Pre-computed analysis features
            "sentiment_label": sentiment_label,
            "sentiment_score": max(sentiment_pos, sentiment_neg, sentiment_neu),
            "sentiment_positive": sentiment_pos,
            "sentiment_negative": sentiment_neg,
            "sentiment_neutral": sentiment_neu,
            "toxicity_label": "toxic" if toxicity > 0.5 else "non-toxic",
            "toxicity_score": toxicity,
            "regard_positive": regard_pos,
            "regard_negative": regard_neg,
            "regard_neutral": regard_neu,
            "regard_other": 0.0,
            "regard_label": regard_label,
            "word_count": len(response.split()),
            "char_count": len(response),
            "sentence_count": response.count(".") + response.count("!") + response.count("?"),
            "avg_word_length": np.mean([len(w) for w in response.split()]),
            "avg_sentence_length": len(response.split()) / max(response.count("."), 1),
            "unique_word_ratio": len(set(response.lower().split())) / max(len(response.split()), 1),
            "question_count": response.count("?"),
            "exclamation_count": response.count("!"),
            "is_refusal": is_refusal,
            "hedging_count": sum(1 for w in ["might", "perhaps", "possibly", "may", "could"]
                                if w in response.lower()),
            "sentiment_bin": sentiment_bin,
            "toxicity_bin": toxicity_bin,
            "regard_bin": regard_label,
            "length_bin": "short" if word_count < 60 else "long" if word_count > 100 else "medium",
        })

    df = pd.DataFrame(rows)
    print(f"Generated {len(df)} synthetic responses with embedded bias patterns")
    print(f"  Categories: {df['category'].value_counts().to_dict()}")
    print(f"  Embedded biases: sentiment={len(SENTIMENT_BIASES)}, "
          f"toxicity={len(TOXICITY_BIASES)}, regard={len(REGARD_BIASES)}")
    return df


if __name__ == "__main__":
    df = generate_sample_data(500)
    df.to_json("data/sample_responses.json", orient="records", indent=2)
    print(f"\nSaved to data/sample_responses.json")
